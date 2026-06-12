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

enum class det_kind { full, half };

// Flat-packed container where every row has the same fixed width _elem_size.
// Storage: std::vector<ElemT> _data of length size * _elem_size (tightly packed).
//   _data[i*_elem_size .. i*_elem_size+_elem_size-1] = row i data
// Only intended for ElemT = size_t or uint32_t.
// _elem_size is a shared static per specialisation: det_vector<ElemT, Kind>
// instances all carry the same row width.  It is set once (from 0) and never changed.
// The Kind tag generates distinct specialisations with independent _elem_size values,
// allowing full-det and half-det containers to coexist without aliasing.
template<typename ElemT, det_kind Kind = det_kind::full>
class det_vector {
    static_assert(std::is_trivially_copyable_v<ElemT>,
                  "det_vector<ElemT> requires a trivially copyable element type");
public:

    // Non-owning view of a single row obtained via reinterpret_cast from the
    // backing store.  _data[k] is data element k — out-of-bounds for the declared
    // array but backed by the flat allocation (C struct-hack pattern).
    // Constructors are deleted so standalone row objects cannot be created.
    struct row {
        ElemT _data[1];

        row()              = delete;
        row(const row&)    = delete;

        size_t size() const noexcept { return det_vector::_elem_size; }

        ElemT& operator[](size_t k) noexcept       { return _data[k]; }
        const ElemT& operator[](size_t k) const noexcept { return _data[k]; }

        ElemT* data() noexcept             { return _data; }
        const ElemT* data() const noexcept { return _data; }

        ElemT* begin() noexcept             { return _data; }
        ElemT* end()   noexcept             { return _data + size(); }
        const ElemT* begin() const noexcept { return _data; }
        const ElemT* end()   const noexcept { return _data + size(); }

        operator std::vector<ElemT>() const {
            return std::vector<ElemT>(_data, _data + size());
        }

        // In-place copy from vector; row size must match.
        row& operator=(const std::vector<ElemT>& v) {
            assert(v.size() == size());
            std::memcpy(_data, v.data(), size() * sizeof(ElemT));
            return *this;
        }

        // In-place copy from another row; memcpy of full row data.
        row& operator=(const row& other) {
            if (this == &other) return *this;
            std::memcpy(_data, other._data, size() * sizeof(ElemT));
            return *this;
        }

        bool operator==(const std::vector<ElemT>& v) const noexcept {
            if (v.size() != size()) return false;
            return std::memcmp(_data, v.data(), size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const std::vector<ElemT>& v) const noexcept {
            return !(*this == v);
        }
        bool operator==(const row& other) const noexcept {
            return std::memcmp(_data, other._data, size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const row& other) const noexcept {
            return !(*this == other);
        }
        bool operator<(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[k] < v[k]) return true;
                if (_data[k] > v[k]) return false;
            }
            return false;
        }
        bool operator<(const row& other) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[k] < other._data[k]) return true;
                if (_data[k] > other._data[k]) return false;
            }
            return false;
        }
        bool operator>(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = 0; k < n; k++) {
                if (_data[k] > v[k]) return true;
                if (_data[k] < v[k]) return false;
            }
            return false;
        }
        bool operator<=(const std::vector<ElemT>& v) const noexcept { return !(*this > v); }
        bool operator>=(const std::vector<ElemT>& v) const noexcept { return !(*this < v); }
        bool operator>(const row& other) const noexcept { return other < *this; }
        bool operator<=(const row& other) const noexcept { return !(other < *this); }
        bool operator>=(const row& other) const noexcept { return !(*this < other); }

        friend void swap(row& a, row& b) noexcept {
            for (size_t k = 0, n = a.size(); k < n; k++) std::swap(a[k], b[k]);
        }
    };

    using value_type = row;

    // Random-access iterator. _ptr points to _data[0] of the current row.
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
        : _size(n)
    {
        _set_elem_size(v.size());
        _data.resize(n * stride());
        for (size_t i = 0; i < n; i++) _init_row(i, v.data());
    }

    det_vector(const det_vector&) = default;
    det_vector& operator=(const det_vector&) = default;

    det_vector(det_vector&& other) noexcept
        : _data(std::move(other._data)),
          _size(other._size)
    {
        other._data.clear();
        other._size = 0;
    }

    det_vector& operator=(det_vector&& other) noexcept {
        if (this != &other) {
            _data  = std::move(other._data);
            _size  = other._size;
            other._data.clear();
            other._size = 0;
        }
        return *this;
    }

    // --- observers ---

    size_t size()      const noexcept { return _size; }
    size_t elem_size() const noexcept { return _elem_size; }
    bool   empty()     const noexcept { return _size == 0; }

    // Direct access to the flat backing buffer (_size * _elem_size elements).
    std::vector<ElemT>&       flat()        noexcept { return _data; }
    const std::vector<ElemT>& cflat() const noexcept { return _data; }

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

    // Set _size = 0 and release _data storage.
    void clear() noexcept { _data.clear(); _size = 0; }

    // Reserve capacity for at least n rows without changing _size or initialising rows.
    void reserve(size_t n) { _data.reserve(n * stride()); }

    size_t capacity() const noexcept {
        return (_elem_size > 0) ? (_data.capacity() / stride()) : 0;
    }

    // Resize to n rows. _data.size() is always exactly n * _elem_size.
    // New rows (if any) are left uninitialised. Requires _elem_size already set.
    void resize(size_t n) {
        assert(_elem_size != 0);
        _data.resize(n * stride());
        _size = n;
    }

    // Resize to n rows. Sets _elem_size if not yet set (asserts match if already set).
    void resize(size_t n, size_t new_elem_size) {
        _set_elem_size(new_elem_size);
        resize(n);
    }

    // Resize to n rows, filling new rows [_size..n) from v.
    void resize(size_t n, const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        size_t old_size = _size;
        _data.resize(n * stride());
        _size = n;
        for (size_t i = old_size; i < n; i++) _init_row(i, v.data());
    }

    // Replace contents with m rows each initialised to v. Sets _elem_size from v.size().
    void assign(size_t m, const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        _size = m;
        _data.resize(m * stride());
        for (size_t i = 0; i < m; i++) _init_row(i, v.data());
    }

    void push_back(const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, v.data());
        ++_size;
    }

    void push_back(const row& r) {
        _set_elem_size(r.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, r._data);
        ++_size;
    }

    void emplace_back(const std::vector<ElemT>& v) { push_back(v); }
    void emplace_back(const row& r) { push_back(r); }

    // Append a row from an iterator range [first, last).
    template<typename InputIt>
    void emplace_back(InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        _set_elem_size(n);
        _data.resize((_size + 1) * stride());
        std::copy(first, last, _data.data() + _size * stride());
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
            std::copy(src.begin(), src.end(), dst);
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
        _data.resize(_size * stride());
        return iterator{_data.data() + pos * stride(), stride()};
    }

    iterator erase(iterator pos) { return erase(pos, pos + 1); }

private:
    std::vector<ElemT> _data;
    size_t _size = 0;

    inline static size_t _elem_size = 0;

    // Set _elem_size to n. No-op if already n; asserts it was 0 otherwise.
    static void _set_elem_size(size_t n) {
        if (_elem_size == n) return;
        assert(_elem_size == 0);
        _elem_size = n;
    }

    size_t stride() const noexcept { return _elem_size; }

    // Write data for row i from src (length _elem_size).
    void _init_row(size_t i, const ElemT* src) noexcept {
        std::memcpy(_data.data() + i * stride(), src, _elem_size * sizeof(ElemT));
    }
};

// Set c to m rows of width n, releasing any prior storage.
template<typename ElemT, det_kind Kind>
inline void clear_and_resize(det_vector<ElemT, Kind>& c, size_t m, size_t n) {
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
