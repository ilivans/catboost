#include "modes.h"
#include "bind_options.h"

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/algo/quantization.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model_build_helper.h>
#include <library/getopt/small/last_getopt.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/preprocess.h>

#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/quantization.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/defaults_helper.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/pairs/util.h>

#include <library/json/json_prettifier.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/scope.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>
#include <util/system/hp_timer.h>
#include <util/system/info.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>


TLearnProgress LoadSnapshot(const TString& snapshotPath) {
    CB_ENSURE(NFs::Exists(snapshotPath), "Snapshot file doesn't exist: " << snapshotPath);
    try {
        TLearnProgress learnProgress;
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(snapshotPath, [&](TIFStream* in) {
            TRestorableFastRng64 rand(0);
            ::Load(in, rand);
            learnProgress.Load(in, true);
        });
        CATBOOST_INFO_LOG << "Snapshot is loaded from: " << snapshotPath << Endl;
        return learnProgress;
    } catch (...) {
        CATBOOST_ERROR_LOG << "Can't load progress from snapshot file: " << snapshotPath << Endl;
        throw;
    }
}


TObliviousTrees BuildObliviousTrees(const TLearnProgress& learnProgress, const TCtrHelper& ctrHelper=TCtrHelper(), const TDataset& learnData=TDataset()) {
    try {
        TObliviousTreeBuilder builder(learnProgress.FloatFeatures, learnProgress.CatFeatures,
                                      learnProgress.ApproxDimension);
        for (size_t treeId = 0; treeId < learnProgress.TreeStruct.size(); ++treeId) {
            TVector <TModelSplit> modelSplits;
            for (const auto &split : learnProgress.TreeStruct[treeId].Splits) {
                auto modelSplit = split.GetModelSplit(learnProgress, ctrHelper, learnData);
                modelSplits.push_back(modelSplit);
            }
            builder.AddTree(modelSplits, learnProgress.LeafValues[treeId],
                            learnProgress.TreeStats[treeId].LeafWeightsSum);
        }
        TObliviousTrees obliviousTrees = builder.Build();
        CATBOOST_DEBUG_LOG << "TObliviousTrees is built" << Endl;
        return obliviousTrees;
    } catch (...) {
        CATBOOST_ERROR_LOG << "Can't build TObliviousTrees" << Endl;
        throw;
    }
}


int mode_snapshot_to_model(int argc, const char* argv[]) {
    NCatboostOptions::TPoolLoadParams poolLoadOptions;
    TString paramsFile;
    NJson::TJsonValue catBoostFlatJsonOptions;
    ParseCommandLine(argc, argv, &catBoostFlatJsonOptions, &paramsFile, &poolLoadOptions);
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputOptionsJson;
    NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson);

    poolLoadOptions.IgnoredFeatures = GetOptionIgnoredFeatures(catBoostJsonOptions);
    auto taskType = NCatboostOptions::GetTaskType(catBoostJsonOptions);
    NCatboostOptions::TOutputFilesOptions outputOptions(taskType);
    if (!outputOptionsJson.Has("train_dir")) {
        outputOptionsJson["train_dir"] = ".";
    }
    outputOptions.Load(outputOptionsJson);

    TLearnProgress learnProgress = LoadSnapshot(outputOptions.GetSnapshotFilename());
    TObliviousTrees obliviousTrees;
    TFullModel Model;

    if (learnProgress.CatFeatures.size()) {
        poolLoadOptions.Validate(taskType);  // Learn dataset path is required

        NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(learnProgress.SerializedTrainParams));
        auto targetConverter = NCB::MakeTargetConverter(updatedParams);
        TProfileInfo profile;
        TTrainPools pools_;
        LoadPools(
                poolLoadOptions,
                1,
                &targetConverter,
                &profile,
                &pools_
        );

        auto pools = TClearablePoolPtrs(pools_, true);
        auto sortedCatFeatures = ToUnsigned(pools.Learn->CatFeatures);
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());
        const int featureCount = pools.Learn->GetFactorCount();
        TLearnContext ctx(
                updatedParams,
                Nothing(),
                Nothing(),
                outputOptions,
                featureCount,
                sortedCatFeatures,
                pools.Learn->FeatureId
        );
        ctx.LearnProgress = std::move(learnProgress);


        TVector<ui64> indices(pools.Learn->Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);
        ui64 minTimestamp = *MinElement(pools.Learn->Docs.Timestamp.begin(), pools.Learn->Docs.Timestamp.end());
        ui64 maxTimestamp = *MaxElement(pools.Learn->Docs.Timestamp.begin(), pools.Learn->Docs.Timestamp.end());
        if (minTimestamp != maxTimestamp) {
            indices = CreateOrderByKey(pools.Learn->Docs.Timestamp);
            ctx.Params.DataProcessingOptions->HasTimeFlag = true;
        }
        if (!ctx.Params.DataProcessingOptions->HasTimeFlag) {
            Shuffle(pools.Learn->Docs.QueryId, ctx.Rand, &indices);
        }


        ELossFunction lossFunction = ctx.Params.LossFunctionDescription.Get().GetLossFunction();
        if (IsPairLogit(lossFunction) && pools.Learn->Pairs.empty()) {
            CB_ENSURE(
                    !pools.Learn->Docs.Target.empty(),
                    "Pool labels are not provided. Cannot generate pairs."
            );
            CATBOOST_WARNING_LOG << "No pairs provided for learn dataset. "
                                 << "Trying to generate pairs using dataset labels." << Endl;
            pools.Learn->Pairs.clear();
            GeneratePairLogitPairs(
                    pools.Learn->Docs.QueryId,
                    pools.Learn->Docs.Target,
                    NCatboostOptions::GetMaxPairCount(ctx.Params.LossFunctionDescription),
                    &ctx.Rand,
                    &(pools.Learn->Pairs));
            CATBOOST_INFO_LOG << "Generated " << pools.Learn->Pairs.size() << " pairs for learn pool." << Endl;
        }

        ApplyPermutation(InvertPermutation(indices), pools.Learn, &ctx.LocalExecutor);
        Y_DEFER {
                ApplyPermutation(indices, pools.Learn, &ctx.LocalExecutor);
        };

        TDataset learnData = BuildDataset(*pools.Learn);

        const TVector<float>& classWeights = ctx.Params.DataProcessingOptions->ClassWeights;
        const auto& labelConverter = ctx.LearnProgress.LabelConverter;
        Preprocess(ctx.Params.LossFunctionDescription, classWeights, labelConverter, learnData);
        CheckLearnConsistency(ctx.Params.LossFunctionDescription, ctx.Params.DataProcessingOptions->AllowConstLabel.Get(), learnData);

        CATBOOST_DEBUG_LOG << "IsQuantized " << pools.Learn->IsQuantized() << Endl;
        if (!pools.Learn->IsQuantized()) {
            GenerateBorders(*pools.Learn, &ctx, &ctx.LearnProgress.FloatFeatures);
        } else {
            ctx.LearnProgress.FloatFeatures = pools.Learn->FloatFeatures;
        }

        for (const TPool* testPoolPtr : pools.Test) {
            const TPool& testPool = *testPoolPtr;
            if (testPool.Docs.GetDocCount() == 0) {
                continue;
            }
            CB_ENSURE(
                    testPool.GetFactorCount() == pools.Learn->GetFactorCount(),
                    "train pool factors count == " << pools.Learn->GetFactorCount() << " and test pool factors count == " << testPool.GetFactorCount()
            );

            // TODO(akhropov): cast will be removed after switch to new Pool format. MLTOOLS-140.
            auto catFeaturesTest = ToUnsigned(testPool.CatFeatures);
            Sort(catFeaturesTest.begin(), catFeaturesTest.end());
            CB_ENSURE(sortedCatFeatures == catFeaturesTest, "Cat features in train and test should be the same.");
        }

        TVector<TDataset> testDatasets;
        for (const TPool* testPoolPtr : pools.Test) {
            testDatasets.push_back(BuildDataset(*testPoolPtr));
            auto& pairs = testDatasets.back().Pairs;
            if (IsPairLogit(lossFunction) && pairs.empty()) {
                GeneratePairLogitPairs(
                        testDatasets.back().QueryId,
                        testDatasets.back().Target,
                        NCatboostOptions::GetMaxPairCount(ctx.Params.LossFunctionDescription),
                        &ctx.Rand,
                        &pairs);
                CATBOOST_INFO_LOG << "Generated " << pairs.size()
                                  << " pairs for test pool " <<  testDatasets.size() << "." << Endl;
            }
        }

        const TDatasetPtrs& testDataPtrs = GetConstPointers(testDatasets);

        for (TDataset& testData : testDatasets) {
            UpdateQueryInfo(&testData);
        }
        for (TDataset& testData : testDatasets) {
            Preprocess(ctx.Params.LossFunctionDescription, classWeights, labelConverter, testData);
            CheckTestConsistency(ctx.Params.LossFunctionDescription, learnData, testData);
        }


        const auto& catFeatureParams = ctx.Params.CatFeatureParams.Get();
//        TVector<TDataset> testDatasets;
        QuantizeTrainPools(
                pools,
                ctx.LearnProgress.FloatFeatures,
                Nothing(),
                ctx.Params.DataProcessingOptions->IgnoredFeatures,
                catFeatureParams.OneHotMaxSize,
                ctx.LocalExecutor,
                &learnData,
                &testDatasets
        );
        ctx.InitContext(learnData, testDataPtrs);

//        ctx.LearnProgress.CatFeatures.resize(sortedCatFeatures.size());
//        for (size_t i = 0; i < sortedCatFeatures.size(); ++i) {
//            auto& catFeature = ctx.LearnProgress.CatFeatures[i];
//            catFeature.FeatureIndex = i;
//            catFeature.FlatFeatureIndex = sortedCatFeatures[i];
//            if (catFeature.FlatFeatureIndex < pools.Learn->FeatureId.ysize()) {
//                catFeature.FeatureId = pools.Learn->FeatureId[catFeature.FlatFeatureIndex];
//            }
//        }
        const auto& systemOptions = ctx.Params.SystemOptions;
        if (!systemOptions->IsSingleHost()) { // send target, weights, baseline (if present), binarized features to workers and ask them to create plain folds
            InitializeMaster(&ctx);
            CB_ENSURE(IsPlainMode(ctx.Params.BoostingOptions->BoostingType), "Distributed training requires plain boosting");
            CB_ENSURE(pools.Learn->CatFeatures.empty(), "Distributed training requires all numeric data");
            MapBuildPlainFold(learnData, &ctx);
        }

//        obliviousTrees = BuildObliviousTrees(ctx.LearnProgress, ctx.CtrsHelper, learnData);
        THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
        try {
            TObliviousTreeBuilder builder(ctx.LearnProgress.FloatFeatures, ctx.LearnProgress.CatFeatures,
                                          ctx.LearnProgress.ApproxDimension);
            for (size_t treeId = 0; treeId < ctx.LearnProgress.TreeStruct.size(); ++treeId) {
                TVector <TModelSplit> modelSplits;
                for (const auto &split : ctx.LearnProgress.TreeStruct[treeId].Splits) {
                    auto modelSplit = split.GetModelSplit(ctx.LearnProgress, ctx.CtrsHelper, learnData);
                    modelSplits.push_back(modelSplit);
                    if (modelSplit.Type == ESplitType::OnlineCtr) {
                        featureCombinationToProjectionMap[modelSplit.OnlineCtr.Ctr.Base.Projection] = split.Ctr.Projection;
                    }
                }
                builder.AddTree(modelSplits, ctx.LearnProgress.LeafValues[treeId],
                                ctx.LearnProgress.TreeStats[treeId].LeafWeightsSum);
            }
            obliviousTrees = builder.Build();
            CATBOOST_DEBUG_LOG << "TObliviousTrees is built" << Endl;
        } catch (...) {
            CATBOOST_ERROR_LOG << "Can't build TObliviousTrees" << Endl;
            throw;
        }



        NCB::TCoreModelToFullModelConverter coreModelToFullModelConverter(
                1,
                ctx.OutputOptions.GetFinalCtrComputationMode(),
                ParseMemorySizeDescription(ctx.Params.SystemOptions->CpuUsedRamLimit.Get()),
                ctx.Params.CatFeatureParams->CtrLeafCountLimit,
                ctx.Params.CatFeatureParams->StoreAllSimpleCtrs,
                catFeatureParams
        );

//        const TDatasetPtrs& testDataPtrs = GetConstPointers(testDatasets);
        TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
        datasetDataForFinalCtrs.LearnData = &learnData;
        datasetDataForFinalCtrs.TestDataPtrs = &testDataPtrs;
        datasetDataForFinalCtrs.LearnPermutation = &ctx.LearnProgress.AveragingFold.LearnPermutation;
        datasetDataForFinalCtrs.Targets = &pools.Learn->Docs.Target;
        datasetDataForFinalCtrs.LearnTargetClass = &ctx.LearnProgress.AveragingFold.LearnTargetClass;
        datasetDataForFinalCtrs.TargetClassesCount = &ctx.LearnProgress.AveragingFold.TargetClassesCount;

        coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
                datasetDataForFinalCtrs,
                featureCombinationToProjectionMap
        );

        Model.ObliviousTrees = std::move(obliviousTrees);
        Model.ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
        if (ctx.OutputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) {
            coreModelToFullModelConverter.WithCoreModelFrom(&Model).Do(
                    &Model,
                    ctx.OutputOptions.ExportRequiresStaticCtrProvider()
            );
        }

        TString outputFile = ctx.OutputOptions.CreateResultModelFullPath();
        for (const auto& format : ctx.OutputOptions.GetModelFormats()) {
            ExportModel(Model, outputFile, format, "", ctx.OutputOptions.AddFileFormatExtension(), &pools.Learn->FeatureId, &pools.Learn->CatFeaturesHashToString);
        }
    } else {
        obliviousTrees = BuildObliviousTrees(learnProgress);
        Model.ObliviousTrees = std::move(obliviousTrees);
        Model.ModelInfo["params"] = learnProgress.SerializedTrainParams;
        ExportModel(Model, outputOptions.GetResultModelFilename(), EModelType::CatboostBinary);
    }
    CATBOOST_INFO_LOG << "Model is saved at: " << outputOptions.GetResultModelFilename() << Endl;

    return 0;
}
