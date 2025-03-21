#pragma once
#include <memory>
#include <type_traits>

namespace prx
{
namespace utilities
{

// is_shared_ptr: Taken from:
// https://stackoverflow.com/questions/41853159/how-to-detect-if-a-type-is-shared-ptr-at-compile-time
// This is though of an extension of std::is_pointer<T> to detect shared_ptr(s).
template <class T>
struct is_shared_ptr : std::false_type
{
};

template <class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{
};

template <class T>
struct is_shared_ptr<std::shared_ptr<T> const> : std::true_type
{
};

template <class T>
struct is_shared_ptr<std::shared_ptr<T> volatile> : std::true_type
{
};

template <class T>
struct is_unique_ptr : std::false_type
{
};

template <class T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type
{
};

template <class T>
struct is_unique_ptr<std::unique_ptr<T> const> : std::true_type
{
};

template <class T>
struct is_unique_ptr<std::unique_ptr<T> volatile> : std::true_type
{
};

template <class T>
struct is_any_ptr : std::integral_constant<bool,
                                           is_shared_ptr<T>::value         // no-lint
                                               || is_unique_ptr<T>::value  // no-lint
                                               || std::is_pointer<T>::value>
{
};

template <typename T, typename = void>
struct is_iterable : std::false_type
{
};

// this gets used only when we can call std::begin() and std::end() on that type
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
  : std::true_type
{
};

template <typename T, typename = void>
struct is_streamable : std::false_type
{
};

template <typename T>
struct is_streamable<T, std::enable_if_t<std::is_convertible_v<
                            decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>>>
  : std::true_type
{
};
}  // namespace utilities
}  // namespace prx
