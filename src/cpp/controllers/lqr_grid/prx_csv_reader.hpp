#pragma once
#include <fstream>
#include <filesystem>
    
// #include "prx/utilities/defs.hpp"
#include "type_conversions.hpp"

namespace prx
{
namespace utilities
{
template <typename T>
static std::vector<T> split(const std::string str)
{

  std::vector<T> result;
  std::istringstream ss(str);
  std::string token;
  while (std::getline(ss, token, ' '))
  {
    if (token.size() > 0)
    {
      std::istringstream ti(token);
      T x;
      if ((ti >> x))
        result.push_back(x);
    }
  }
  return result;
}

class csv_reader_t
{
public:
  template <typename T>
  using Line = std::vector<T>;
  template <typename T>
  using Block = Line<Line<T>>;  // A block is a 'line' of lines
  // member typedefs provided through inheriting from std::iterator
  template <typename T = std::string>
  class iterator
  {
    using iterator_category = std::output_iterator_tag;
    using value_type = Line<T>;  // crap
    using difference_type = Line<T>;
    using pointer = const Line<T>*;
    using reference = Line<T>;
    csv_reader_t* _ptr;
    Line<T> _line;

  public:
    explicit iterator(csv_reader_t* ptr) : _ptr(ptr), _line()
    {
      if (_ptr != nullptr)
      {
        if (_ptr->has_next_line())
          _line = _ptr->next_line<T>();
      }
    }

    iterator& operator++()
    {
      if (_ptr != nullptr && _ptr->has_next_line())
      {
        _line = _ptr->next_line<T>();
      }
      else
      {
        _ptr = nullptr;
        _line = Line<T>();
      }
      return *this;
    }
    iterator operator++(int)
    {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(iterator other) const
    {
      return _ptr == other._ptr && _line == other._line;
    }
    bool operator!=(iterator other) const
    {
      return !(*this == other);
    }
    reference operator*() const
    {
      return _line;
    }
  };
  template <typename T = std::string>
  iterator<T> begin()
  {
    return iterator(this);
  }

  template <typename T = std::string>
  iterator<T> end()
  {
    return iterator(nullptr);
  }

  csv_reader_t() = delete;
  csv_reader_t(const std::string filename, const char separator = ' ')
    : _filename(filename), file(filename.c_str())
  {
    assert(std::filesystem::exists(filename));
  }

  ~csv_reader_t()
  {
  }

  inline bool has_next_line() const
  {
    return !file.eof();
  }
  inline bool is_open() const
  {
    return file.is_open();
  }

  template <typename T = std::string>
  Line<T> next_line()
  {
    std::string line;
    std::getline(file, line);
    return split<T>(line);
  }

  template <typename T = std::string, typename Function>
  Line<T> next_line(Function f)
  {
    Line<T> line;
    do
    {
      line = next_line<T>();
      if (f(line))
      {
        return line;
      }
    } while (has_next_line());
    return Line<T>();
  }

  /**
   * @brief      Get the next line that satisfies line[idx] == value
   *
   * @param[in]  value     Value
   * @param[in]  idx   The index
   *
   * @return     Line that satisfies "line[idx] == value"
   */
  template <typename T>
  Line<T> next_line(const T& value, const std::size_t idx)
  {
    return next_line<T>([&](const Line<T>& line) { return line.size() > idx && line[idx] == value; });
  }

  template <typename T = std::string>
  Block<T> next_block()
  {
    Line<T> line;
    Block<T> block;

    while (has_next_line())
    {
      line = next_line<T>();
      if (line.size() == 0)
        break;
      block.emplace_back(line);
    }
    return block;
  }

  template <typename T>
  Line<T> read_column(const std::size_t idx)
  {
    std::vector<std::size_t> col = { idx };
    return read_columns<T>(col)[0];
  }

  template <typename T, typename Container>
  Block<T> read_columns(const Container container)
  {
    Block<T> columns(container.size());
    while (has_next_line())
    {
      auto line = next_line<std::string>();
      for (int i = 0; i < container.size(); ++i)
      {
        const std::size_t idx{ container[i] };
        if (line.size() < idx)
          continue;
        columns[i].push_back(prx::utilities::convert_to<T>(line[idx]));
      }
    }
    return columns;
  }

private:
  const std::string _filename;
  std::ifstream file;
};

}  // namespace utilities
}  // namespace prx
