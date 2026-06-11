/// This is a part of qscd
/**
@file det_vector.h
@brief flat-packed fixed-row-width replacement for std::vector<std::vector<ElemT>>
 */

#ifndef SBD_FRAMEWORK_DET_VECTOR_H
#define SBD_FRAMEWORK_DET_VECTOR_H

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <vector>

namespace sbd {

// Flat-packed container where every row has the same fixed width _elem_size.
// Storage: std::vector<ElemT> _data of length size * (_elem_size+1).
//   _data[i*(_elem_size+1)]               = _elem_size  (row header)
//   _data[i*(_elem_size+1)+1 .. +_elem_size] = row i data
// Only intended for ElemT = size_t or uint32_t.
template<typename ElemT>
class det_vector {
    static_assert(std::is_trivially_copyable_v<ElemT>,
                  "det_vector<ElemT> requires a trivially copyable element type");
public:

    // Non-owning view of a single row obtained via reinterpret_cast from the
    // backing store.  _data[0] is the header (= elem_size); data elements are
    // at _data[1+k] — out-of-bounds for the declared array but backed by the
    // flat allocation (C struct-hack pattern).
    // Constructors are deleted so standalone row objects cannot be created.
    struct row {
        ElemT _data[1];

        row()              = delete;
        row(const row&)    = delete;

        size_t size() const noexcept { return static_cast<size_t>(_data[0]); }

        ElemT& operator[](size_t k) noexcept       { return _data[1 + k]; }
        const ElemT& operator[](size_t k) const noexcept { return _data[1 + k]; }

        ElemT* data() noexcept             { return _data + 1; }
        const ElemT* data() const noexcept { return _data + 1; }

        ElemT* begin() noexcept             { return _data + 1; }
        ElemT* end()   noexcept             { return _data + 1 + size(); }
        const ElemT* begin() const noexcept { return _data + 1; }
        const ElemT* end()   const noexcept { return _data + 1 + size(); }

        operator std::vector<ElemT>() const {
            return std::vector<ElemT>(_data + 1, _data + 1 + size());
        }

        // In-place copy from vector; row size must match.
        row& operator=(const std::vector<ElemT>& v) {
            assert(v.size() == size());
            std::memcpy(_data + 1, v.data(), size() * sizeof(ElemT));
            return *this;
        }

        // In-place copy from another row; memcpy of full row data.
        row& operator=(const row& other) {
            if (this == &other) return *this;
            assert(other.size() == size());
            std::memcpy(_data + 1, other._data + 1, size() * sizeof(ElemT));
            return *this;
        }

        bool operator==(const std::vector<ElemT>& v) const noexcept {
            if (v.size() != size()) return false;
            return std::memcmp(_data + 1, v.data(), size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const std::vector<ElemT>& v) const noexcept {
            return !(*this == v);
        }
        bool operator==(const row& other) const noexcept {
            if (other.size() != size()) return false;
            return std::memcmp(_data + 1, other._data + 1, size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const row& other) const noexcept {
            return !(*this == other);
        }
        bool operator<(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[1 + k] < v[k]) return true;
                if (_data[1 + k] > v[k]) return false;
            }
            return false;
        }
        bool operator<(const row& other) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[1 + k] < other._data[1 + k]) return true;
                if (_data[1 + k] > other._data[1 + k]) return false;
            }
            return false;
        }
        bool operator>(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[1 + k] > v[k]) return true;
                if (_data[1 + k] < v[k]) return false;
            }
            return false;
        }
        bool operator<=(const std::vector<ElemT>& v) const noexcept { return !(*this > v); }
        bool operator>=(const std::vector<ElemT>& v) const noexcept { return !(*this < v); }
        bool operator>(const row& other) const noexcept { return other < *this; }
        bool operator<=(const row& other) const noexcept { return !(other < *this); }
        bool operator>=(const row& other) const noexcept { return !(*this < other); }

        friend void swap(row& a, row& b) noexcept {
            assert(a.size() == b.size());
            for (size_t k = 0, n = a.size(); k < n; k++) std::swap(a[k], b[k]);
        }
    };

    using value_type = row;

    // Random-access iterator. _ptr points to _data[0] of the current row (the header).
    struct iterator {
        using value_type        = row;
        using reference         = row&;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
        using pointer           = row*;

        ElemT*  _ptr;
        size_t  _stride;

        row& operator*()  const noexcept {
            return *reinterpret_cast<row*>(_ptr);
        }
        row& operator[](difference_type n) const noexcept {
            return *reinterpret_cast<row*>(
                _ptr + n * static_cast<difference_type>(_stride));
        }
        row* operator->() const noexcept {
            return reinterpret_cast<row*>(_ptr);
        }

        iterator& operator++() noexcept { _ptr += _stride; return *this; }
        iterator  operator++(int) noexcept { auto t = *this; _ptr += _stride; return t; }
        iterator& operator--() noexcept { _ptr -= _stride; return *this; }
        iterator  operator--(int) noexcept { auto t = *this; _ptr -= _stride; return t; }
        iterator& operator+=(difference_type n) noexcept {
            _ptr += n * static_cast<difference_type>(_stride); return *this;
        }
        iterator& operator-=(difference_type n) noexcept {
            _ptr -= n * static_cast<difference_type>(_stride); return *this;
        }
        iterator operator+(difference_type n) const noexcept {
            return {_ptr + n * static_cast<difference_type>(_stride), _stride};
        }
        iterator operator-(difference_type n) const noexcept {
            return {_ptr - n * static_cast<difference_type>(_stride), _stride};
        }
        difference_type operator-(const iterator& o) const noexcept {
            return (_ptr - o._ptr) / static_cast<difference_type>(_stride);
        }
        friend iterator operator+(difference_type n, const iterator& it) noexcept {
            return it + n;
        }

        bool operator==(const iterator& o) const noexcept { return _ptr == o._ptr; }
        bool operator!=(const iterator& o) const noexcept { return _ptr != o._ptr; }
        bool operator< (const iterator& o) const noexcept { return _ptr <  o._ptr; }
        bool operator> (const iterator& o) const noexcept { return _ptr >  o._ptr; }
        bool operator<=(const iterator& o) const noexcept { return _ptr <= o._ptr; }
        bool operator>=(const iterator& o) const noexcept { return _ptr >= o._ptr; }
    };

    // --- construction ---

    det_vector() = default;
    det_vector(size_t n, size_t elem_size) { resize(n, elem_size); }

    // Construct n rows, each a copy of v. Sets _elem_size from v.size().
    det_vector(size_t n, const std::vector<ElemT>& v)
        : _elem_size(v.size()), _size(n), _data(n * stride())
    {
        for (size_t i = 0; i < n; i++) _init_row(i, v.data());
    }

    det_vector(const det_vector&) = default;
    det_vector& operator=(const det_vector&) = default;

    det_vector(det_vector&& other) noexcept
        : _data(std::move(other._data)),
          _elem_size(other._elem_size),
          _size(other._size)
    {
        other._data.clear();
        other._size = 0;
    }

    det_vector& operator=(det_vector&& other) noexcept {
        if (this != &other) {
            _data      = std::move(other._data);
            _elem_size = other._elem_size;
            _size      = other._size;
            other._data.clear();
            other._size = 0;
        }
        return *this;
    }

    // --- observers ---

    size_t size()      const noexcept { return _size; }
    size_t elem_size() const noexcept { return _elem_size; }
    bool   empty()     const noexcept { return _size == 0; }

    // --- element access ---

    row& operator[](size_t i) noexcept {
        return *reinterpret_cast<row*>(_data.data() + i * stride());
    }
    const row& operator[](size_t i) const noexcept {
        return *reinterpret_cast<const row*>(_data.data() + i * stride());
    }

    // --- iterators ---

    iterator begin() noexcept {
        return iterator{_data.data(), stride()};
    }
    iterator end() noexcept {
        return iterator{_data.data() + _size * stride(), stride()};
    }
    // const begin/end return the same iterator type; row& from a const det_vector
    // is the pragmatic research-code tradeoff (no separate const_iterator).
    iterator begin() const {
        return iterator{const_cast<ElemT*>(_data.data()), stride()};
    }
    iterator end() const {
        return iterator{const_cast<ElemT*>(_data.data()) + _size * stride(), stride()};
    }

    // --- modifiers ---

    // Set _size = 0; keep _data allocated so a subsequent grow avoids reallocation.
    void clear() noexcept { _size = 0; }

    // Reserve capacity for at least n rows without changing _size or initialising rows.
    void reserve(size_t n) {
        if (n > capacity()) _data.reserve(n * stride());
    }

    size_t capacity() const noexcept {
        return (_elem_size > 0) ? (_data.capacity() / stride()) : 0;
    }

    // Resize to n rows.
    // - n <= allocated: only updates _size; headers beyond _size remain valid.
    // - n >  allocated: extends _data and initialises only the new headers.
    // Requires _elem_size already set.
    void resize(size_t n) {
        if (n * stride() > _data.size()) {
            size_t allocated = _data.size() / stride();
            _data.resize(n * stride());
            for (size_t i = allocated; i < n; i++) _init_row(i);
        }
        _size = n;
    }

    // Resize to n rows, setting or changing _elem_size.
    // If new_elem_size != _elem_size, asserts n==0 or _size==0, clears _data, updates stride.
    // resize(0, n) is safe on a non-empty container — use it to clear and change elem_size.
    void resize(size_t n, size_t new_elem_size) {
        if (new_elem_size != _elem_size) {
            assert(n == 0 || _size == 0);
            _elem_size = new_elem_size;
            _data.clear();
        }
        resize(n);
    }

    // Resize to n rows, filling new rows [_size..n) from v.
    // Uses the same no-shrink and elem_size-change logic as the other resize overloads.
    void resize(size_t n, const std::vector<ElemT>& v) {
        if (v.size() != _elem_size) {
            assert(n == 0 || _size == 0);
            _elem_size = v.size();
            _data.clear();
        }
        if (n * stride() > _data.size()) {
            _data.resize(n * stride());
        }
        for (size_t i = _size; i < n; i++) _init_row(i, v.data());
        _size = n;
    }

    // Replace contents with m rows each initialised to v. Sets _elem_size from v.size().
    void assign(size_t m, const std::vector<ElemT>& v) {
        _elem_size = v.size();
        _size = m;
        _data.resize(m * stride());
        for (size_t i = 0; i < m; i++) _init_row(i, v.data());
    }

    void push_back(const std::vector<ElemT>& v) {
        if (_elem_size == 0) _elem_size = v.size();
        assert(_elem_size == v.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, v.data());
        ++_size;
    }

    void push_back(const row& r) {
        if (_elem_size == 0) _elem_size = r.size();
        assert(_elem_size == r.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, r._data + 1);
        ++_size;
    }

    void emplace_back(const std::vector<ElemT>& v) { push_back(v); }
    void emplace_back(const row& r) { push_back(r); }

    // Append a row from an iterator range [first, last).
    template<typename InputIt>
    void emplace_back(InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        if (_elem_size == 0) _elem_size = n;
        assert(_elem_size == n);
        _data.resize((_size + 1) * stride());
        ElemT* hdr = _data.data() + _size * stride();
        hdr[0] = static_cast<ElemT>(_elem_size);
        std::copy(first, last, hdr + 1);
        ++_size;
    }

    // Insert rows from [first, last) before pos. Each source row must have
    // size() == _elem_size. Works for both det_vector::iterator and
    // std::vector<std::vector<ElemT>>::iterator source ranges.
    template<typename InputIt>
    iterator insert(iterator pos, InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        if (n == 0) return pos;
        size_t pos_idx = static_cast<size_t>(pos - begin());
        size_t old_size = _size;
        _data.resize((_size + n) * stride());
        _size += n;
        ElemT* base = _data.data();
        if (pos_idx < old_size) {
            std::memmove(base + (pos_idx + n) * stride(),
                         base + pos_idx * stride(),
                         (old_size - pos_idx) * stride() * sizeof(ElemT));
        }
        ElemT* dst = base + pos_idx * stride();
        for (auto it = first; it != last; ++it, dst += stride()) {
            const auto& src = *it;
            dst[0] = static_cast<ElemT>(_elem_size);
            std::copy(src.begin(), src.end(), dst + 1);
        }
        return iterator{base + pos_idx * stride(), stride()};
    }

    iterator erase(iterator first, iterator last) {
        if (first == last) return last;
        size_t pos  = static_cast<size_t>(first - begin());
        size_t n    = static_cast<size_t>(last  - first);
        size_t tail = _size - (pos + n);
        if (tail > 0)
            std::memmove(_data.data() + pos * stride(),
                         _data.data() + (pos + n) * stride(),
                         tail * stride() * sizeof(ElemT));
        _size -= n;
        return iterator{_data.data() + pos * stride(), stride()};
    }

    iterator erase(iterator pos) { return erase(pos, pos + 1); }

private:
    std::vector<ElemT> _data;
    size_t _elem_size = 0;
    size_t _size      = 0;

    size_t stride() const noexcept { return _elem_size + 1; }

    // Write header for row i; data left uninitialised.
    void _init_row(size_t i) noexcept {
        _data[i * stride()] = static_cast<ElemT>(_elem_size);
    }

    // Write header + data for row i from src (length _elem_size).
    void _init_row(size_t i, const ElemT* src) noexcept {
        ElemT* hdr = _data.data() + i * stride();
        hdr[0] = static_cast<ElemT>(_elem_size);
        std::memcpy(hdr + 1, src, _elem_size * sizeof(ElemT));
    }
};

// Clear c and resize to m rows of width n.
// clear_and_resize(c, 0, n) replaces clear_and_set_elem_size: clears and sets width.
template<typename ElemT>
inline void clear_and_resize(det_vector<ElemT>& c, size_t m, size_t n) {
    c.clear();
    c.resize(m, n);
}
template<typename ElemT>
inline void clear_and_resize(std::vector<std::vector<ElemT>>& c, size_t m, size_t n) {
    c.resize(m);
    for (auto& row : c) row.resize(n);
}

} // namespace sbd

#endif // SBD_FRAMEWORK_DET_VECTOR_H
