#pragma once
#include <type_traits>
#include <sstream>
#include <iomanip>
#include "template_utils.hpp"

namespace prx
{
namespace utilities
{

template <typename StringType, typename ArithmeticType,  // no-lint
          std::enable_if_t<std::is_arithmetic<ArithmeticType>::value && std::is_same<StringType, std::string>::value,
                           bool> = true>
inline StringType convert_to(const ArithmeticType& arithmetic)
{
  // Not using std::to_string because (from https://en.cppreference.com/w/cpp/string/basic_string/to_string):
  // "With floating point types std::to_string may yield unexpected results as the number of significant digits in the
  // returned string can be zero, see the example."
  // Using std::stringstream instead
  std::stringstream strstr;
  strstr << std::fixed << std::setprecision(5) << arithmetic;
  return strstr.str();
}

template <typename T, typename StringType,
          std::enable_if_t<std::is_integral<T>::value && std::is_same<StringType, std::string>::value, bool> = true>
inline T convert_to(const StringType& str)
{
  return std::stoi(str);
}

template <
    typename T, typename StringType,
    std::enable_if_t<std::is_floating_point<T>::value && std::is_same<StringType, std::string>::value, bool> = true>
inline T convert_to(const StringType& str)
{
  return std::stod(str);
}

template <typename To, typename From,
          std::enable_if_t<!std::is_same<To, From>::value && std::is_floating_point<To>::value &&
                               std::is_floating_point<From>::value,
                           bool> = true>
inline To convert_to(const From& value)
{
  return static_cast<To>(value);
}

template <typename To, typename From,
          std::enable_if_t<std::is_floating_point<To>::value && std::is_integral<From>::value, bool> = true>
inline To convert_to(const From& value)
{
  return static_cast<To>(value);
}

template <typename To, typename From, std::enable_if_t<std::is_same<To, From>::value, bool> = true>
inline To convert_to(const From& value)
{
  return value;
}

// TODO: iterable to iterable eg: std::vector<double> to std::vector<string>
template <typename StringType, typename From,
          std::enable_if_t<                                        // no-lint
              is_iterable<From>{}                                  // no-lint
                  && std::is_same<StringType, std::string>::value  // no-lint
                  && not std::is_same<From, std::string>::value,   // Need this because string is iterable
                                                                   // but that case makes no sense here
              bool> = true>
inline StringType convert_to(const From& iterable)
{
  StringType str{};
  for (auto value : iterable)
  {
    str += convert_to<StringType>(value);
    str += " " ;
  }
  return str;
}
}  // namespace utilities
}  // namespace prx
