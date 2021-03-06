#pragma once

#include "fwd.h"

#include <util/memory/alloc.h>

#include <deque>
#include <memory>
#include <initializer_list>

template <class T, class A /*= std::allocator<T>*/>
class TDeque: public std::deque<T, TReboundAllocator<A, T>> {
public:
    using TBase = std::deque<T, TReboundAllocator<A, T>>;
    using TSelf = TDeque<T, A>;
    using allocator_type = typename TBase::allocator_type;
    using size_type = typename TBase::size_type;
    using value_type = typename TBase::value_type;

    inline TDeque()
        : TBase()
    {
    }

    inline TDeque(const typename TBase::allocator_type& a)
        : TBase(a)
    {
    }

    explicit inline TDeque(size_type count)
        : TBase(count)
    {
    }

    inline TDeque(size_type count, const T& val)
        : TBase(count, val)
    {
    }

    template <class TIter>
    inline TDeque(TIter first, TIter last)
        : TBase(first, last)
    {
    }

    inline TDeque(const TSelf& src)
        : TBase(src)
    {
    }

    inline TDeque(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    inline TDeque(std::initializer_list<value_type> il, const allocator_type& alloc = allocator_type())
        : TBase(il, alloc)
    {
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        TBase::operator=(std::forward<TSelf>(src));
        return *this;
    }

    inline size_type operator+() const noexcept {
        return this->size();
    }

    inline yssize_t ysize() const noexcept {
        return (yssize_t)this->size();
    }

    Y_PURE_FUNCTION
    inline bool empty() const noexcept {
        return TBase::empty();
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
