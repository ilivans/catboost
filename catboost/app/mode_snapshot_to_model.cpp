#include "modes.h"
#include "bind_options.h"

#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/quantization.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/pairs/util.h>
#include <catboost/libs/train_lib/train_model.h>
#include <util/generic/scope.h>

//#include <catboost/libs/options/metric_options.h>


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


void BuildModelWithCatFeatures(TLearnProgress* learnProgress,
                               const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                               const NCatboostOptions::TOutputFilesOptions& outputOptions) {
    poolLoadOptions.Validate(ETaskType::CPU);  // Learn dataset path is required

    NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(learnProgress->SerializedTrainParams));
    auto targetConverter = NCB::MakeTargetConverter(updatedParams);
    TProfileInfo profile;
    TTrainPools train_pools;
    LoadPools(
            poolLoadOptions,
            1,
            &targetConverter,
            &profile,
            &train_pools
    );

    auto pools = TClearablePoolPtrs(train_pools, true);
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
    ctx.LearnProgress = std::move(*learnProgress);

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
    ApplyPermutation(InvertPermutation(indices), pools.Learn, &ctx.LocalExecutor);
    Y_DEFER {
            ApplyPermutation(indices, pools.Learn, &ctx.LocalExecutor);
    };
    TDataset learnData = BuildDataset(*pools.Learn);

    ELossFunction lossFunction = ctx.Params.LossFunctionDescription.Get().GetLossFunction();
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

//    const bool isMulticlass = IsMultiClass(lossFunction, ctx.Params.MetricOptions);
//
//    if (isMulticlass) {
//        int classesCount = GetClassesCount(
//                ctx.Params.DataProcessingOptions->ClassesCount,
//                ctx.Params.DataProcessingOptions->ClassNames
//        );
//        ctx.LearnProgress.LabelConverter.Initialize(learnData.Target, classesCount);
//        ctx.LearnProgress.ApproxDimension = ctx.LearnProgress.LabelConverter.GetApproxDimension();
//    }

    const auto& catFeatureParams = ctx.Params.CatFeatureParams.Get();
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

    THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
    TObliviousTrees obliviousTrees;
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

    TFullModel model;
    model.ObliviousTrees = std::move(obliviousTrees);
    model.ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
    if (ctx.OutputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) {
        coreModelToFullModelConverter.WithCoreModelFrom(&model).Do(
                &model,
                ctx.OutputOptions.ExportRequiresStaticCtrProvider()
        );
    }
//    CB_ENSURE(isMulticlass == ctx.LearnProgress.LabelConverter.IsInitialized(),
//              "LabelConverter must be initialized ONLY for multiclass problem");
//    if (isMulticlass) {
//        model.ModelInfo["multiclass_params"] = ctx.LearnProgress.LabelConverter.SerializeMulticlassParams(
//                ctx.Params.DataProcessingOptions->ClassesCount,
//                ctx.Params.DataProcessingOptions->ClassNames
//        );;
//    }

    TString outputFile = ctx.OutputOptions.CreateResultModelFullPath();
    for (const auto& format : ctx.OutputOptions.GetModelFormats()) {
        ExportModel(model, outputFile, format, "", ctx.OutputOptions.AddFileFormatExtension(), &pools.Learn->FeatureId, &pools.Learn->CatFeaturesHashToString);
    }
    CATBOOST_INFO_LOG << "Model is saved at: " << ctx.OutputOptions.CreateResultModelFullPath() << Endl;
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
    NCatboostOptions::TOutputFilesOptions outputOptions(ETaskType::CPU);
    if (!outputOptionsJson.Has("train_dir")) {
        outputOptionsJson["train_dir"] = ".";
    }
    outputOptions.Load(outputOptionsJson);

    TLearnProgress learnProgress = LoadSnapshot(outputOptions.GetSnapshotFilename());
    if (learnProgress.CatFeatures.size()) {
        BuildModelWithCatFeatures(&learnProgress, poolLoadOptions, outputOptions);
    } else {
        TObliviousTrees obliviousTrees = BuildObliviousTrees(learnProgress);
        TFullModel model;
        model.ObliviousTrees = std::move(obliviousTrees);
        model.ModelInfo["params"] = learnProgress.SerializedTrainParams;

        TString outputFile = outputOptions.CreateResultModelFullPath();
        for (const auto& format : outputOptions.GetModelFormats()) {
            ExportModel(model, outputFile, format, "", outputOptions.AddFileFormatExtension());
        }
        CATBOOST_INFO_LOG << "Model is saved at: " << outputOptions.CreateResultModelFullPath() << Endl;
    }

    return 0;
}
