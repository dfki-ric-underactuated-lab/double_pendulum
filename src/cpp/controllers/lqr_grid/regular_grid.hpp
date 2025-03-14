#pragma once

// #include "prx/utilities/defs.hpp"
#include "prx_csv_reader.hpp"

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <Eigen/Dense>

namespace prx
{
namespace utilities
{

/**
 *
 * @brief <b> A regular grid. </b>
 *
 * @author Edgar Granados
 */

template <typename ElementIn, int Dimension>
class cube_cell_t
{
  cube_cell_t() {};

public:
  using Element = ElementIn;
  using Coordinate = Eigen::Vector<double, Dimension>;
  using CoordinatePtr = std::shared_ptr<Coordinate>;

  // regular_grid_t(const double min, const double max, const std::size_t total_cells)
  cube_cell_t(const Coordinate& min_vertex, const CoordinatePtr& lengths) : _min_vertex(min_vertex), _lengths(lengths)
  {
  }
  cube_cell_t(const Coordinate& min_vertex, const Coordinate& lengths)
    : _min_vertex(min_vertex), _lengths(std::make_shared<Coordinate>(lengths))
  {
  }

  Element& element()
  {
    return _element;
  }

  // Easiest to think in binary numbers. Total corners = Dim x Dim, vertex 3 = 11;
  Coordinate vertex(const int i)
  {
    Eigen::Array<bool, Dimension, 1> mask{ Eigen::Array<bool, Dimension, 1>::Zero() };
    int bit{ 1 };
    for (int j = 0; j < Dimension; ++j)
    {
      mask[j] = bit & i;
      bit = bit << 1;
    }
    const Coordinate vertex{ mask.template cast<double>() * (*_lengths).array() };
    const Coordinate ci{ _min_vertex + vertex };
    return ci;
  }

  std::vector<Coordinate> vertices()
  {
    std::vector<Coordinate> all_vertices;
    for (int i = 0; i < Dimension * Dimension; ++i)
    {
      all_vertices.emplace_back(vertex(i));
    }
    return all_vertices;
  }

protected:
  Element _element;
  const Coordinate _min_vertex;
  const CoordinatePtr _lengths;
};

template <typename CellType, std::size_t Dimension>
class regular_grid_t
{
public:
  using Grid = regular_grid_t<CellType, Dimension>;
  using CellPtr = std::shared_ptr<CellType>;
  using Coordinate = Eigen::Vector<double, Dimension>;
  using CoordinatePtr = std::shared_ptr<Coordinate>;
  using IntCoord = Eigen::Vector<int, Dimension>;
  using Iterator = typename std::vector<CellPtr>::iterator;
  using EigenArray = Eigen::Array<double, Dimension, 1>;
  using Element = typename CellType::Element;
private:
  regular_grid_t() {};

  bool step(Coordinate& x)
  {
    for (int i = 0; i < Dimension; ++i)
    {
      const double& step{ (*_cell_length)[i] };
      x[i] = x[i] + step;
      const double diff{ x[i] - _max[i] };
      // PRX_DBG_VARS(x.transpose(), _max.transpose(), x[i] <= _max[i]);
      // PRX_DBG_VARS(x[i], _max[i], diff, diff < step * step);
      // std::cout << std::setprecision(56) << " diff: " << diff << std::setprecision(prx::constants::precision) <<
      // "\n"; if (x[i] < _max[i]) if (std::abs(diff) > 1e-5)
      if (x[i] + 1e-5 < _max[i])
      {
        return true;
      }
      x[i] = _min[i];
    }
    return false;
  }

  void allocate_all()
  {
    // const Eigen::Vector<int, Dimension> cells{ _total_cells + Eigen::Vector<int, Dimension>::Ones() };
    // const int total{ cells.prod() };
    const int total{ _total_cells.prod() };

    // PRX_DBG_VARS(_total_cells.transpose(), total);
    _grid.resize(total);
    Coordinate x{ _min };
    int i{ 0 };
    do
    {
      const std::size_t idx{ index(x) };
      _grid[idx] = std::make_shared<CellType>(x, _cell_length);
      // PRX_DBG_VARS(i, idx, x.transpose());
      const Coordinate v0{ _grid[idx]->vertex(0) };
      // PRX_DBG_VARS(x.transpose(), v0.transpose());
      // _grid.emplace_back(new CellType(x, _cell_length));
      i++;
    } while (step(x));
  }

  static IntCoord compute_total(const Coordinate min, const Coordinate max, const Coordinate cell_info)
  {
    IntCoord res{ IntCoord::Zero() };
    for (int i = 0; i < Dimension; ++i)
    {
      const double diff{ max[i] - min[i] };
      res[i] = diff / static_cast<double>(cell_info[i]);
      // PRX_DBG_VARS(diff, res[i], max[i], cell_info[i]);
      // auto dv = std::div(static_cast diff, cell_info[i]);
      // PRX_DBG_VARS(dv.quot, dv.rem);
      // res[i] = dv.quot;
      const double rem{ std::abs(std::remainder(diff, static_cast<double>(cell_info[i]))) };
      // PRX_DBG_VARS(diff, rem, res[i], cell_info[i]);
      if (rem > 0.001)
      {
        res[i]++;
      }
    }

    // PRX_DBG_VARS(res.transpose());
    return res;
  }

  static Coordinate init_2pow(Eigen::Vector<int, Dimension> total_cells)
  {
    Coordinate res{ Coordinate::Ones() };

    double accum{ 1.0 };
    for (std::size_t i = 0; i < Dimension; ++i)
    {
      res[i] = accum;
      accum *= total_cells[i];
    }
    // PRX_DBG_VARS(res);
    return res;
  }

public:
  
  regular_grid_t(const std::string filename) : regular_grid_t(from_file(filename))
  {
  }

  regular_grid_t(const Coordinate min, const Coordinate max, const Coordinate cell_length)
    : _min(min)
    , _max(max)
    , _cell_length(std::make_shared<Coordinate>(cell_length))
    , _cell_length_inv((EigenArray::Ones() / _cell_length->array()).round())
    , _total_cells(compute_total(min, max, cell_length))
    , _2power(init_2pow(_total_cells))
  {
    allocate_all();
  }

  template <typename... Ts>
  std::size_t index(Ts... xs) const
  {
    const Coordinate x{ Coordinate(xs...) };
    std::size_t idx{ 0 };
    const Coordinate xp{ x - _min };

    const Coordinate k{ xp.array() * _cell_length_inv };
    for (int i = 0; i < Dimension; ++i)
    {
      const int ki{ static_cast<int>(std::floor(k[i] + 1e-10)) };

      idx += _2power[i] * ki;
    }

    return idx;
  }

  template <typename... Ts>
  bool in_bounds(Ts... xs)
  {
    const Coordinate x{ Coordinate(xs...) };
    for (int i = 0; i < Dimension; ++i)
    {
      if (not(_min[i] <= x[i] and x[i] < _max[i]))
      {
        return false;
      }
    }
    return true;
  }

  template <typename... Ts>
  CellPtr& at(Ts... xs)
  {
    const Coordinate x{ Coordinate(xs...) };
    return _grid[index(x)];
  }

  template <typename... Ts>
  Element& query(Ts... xs)
  {
    const Coordinate x{ Coordinate(xs...) };
    return _grid[index(x)]->element();
  }

  template <typename... Ts>
  CellPtr& operator()(Ts... xs)
  {
    return at(xs...);
  }

  // ElementToCellFunction: A function that takes as input std::vector<std::string> and returns the corresponding
  // Element Called once per line, where the input vector of strings is the size of the element stored per cell
  static Grid from_file(const std::string filename)
  {
    using prx::utilities::convert_to;
    using prx::utilities::csv_reader_t;
    using Line = std::vector<std::string>;
    csv_reader_t reader_tree(filename);
    reader_tree.next_line();  // Remove first line (header)
    reader_tree.next_line();  // Remove second line (header)
    const Line params{ reader_tree.next_line() };
    const int dim{ convert_to<int>(params[0]) };
    assert(dim == Dimension);
    Coordinate min, max, cell_length;

    // PRX_DBG_VARS(params);
    for (int i = 0; i < Dimension; ++i)
    {
      min[i] = convert_to<double>(params[1 + i]);
      max[i] = convert_to<double>(params[1 + Dimension + i]);
      cell_length[i] = convert_to<double>(params[1 + 2 * Dimension + i]);
    }
    Grid grid(min, max, cell_length);

    Coordinate vertex;
    // typename CellType::Element element;
    while (reader_tree.has_next_line())
    {
      Line line{ reader_tree.next_line() };
      if (line.size() == 0)
        continue;
      // PRX_DBG_VARS(line);
      for (int i = 0; i < Dimension; ++i)
      {
        vertex[i] = convert_to<double>(line[i]) + cell_length[i] / 2.0;
        // element[i] = convert_to<double>(line[Dimension + i]);
      }
      Line subline{ line.begin() + Dimension, line.end() };
      for (int i = 0; i < subline.size(); ++i)
      {
        grid(vertex)->element()[i] = convert_to<double>(subline[i]);
      }
      // grid(vertex)->element() = func(subline);
    }

    return grid;
  }

  template <typename ValidCellFunction>
  void to_file(const std::string filename, ValidCellFunction& func)
  {
    std::ofstream ofs(filename.c_str());
    ofs << "# First line are grid parameters: ";
    ofs << "Dimension min[0] (...) min[Dimension-1] ";
    ofs << "max[0] (...) max[Dimension-1] ";
    ofs << "cell_length[0] (...) cell_length[Dimension-1] \n";
    ofs << "# Remaining lines is the grid as: min_vertex[0] (...) min_vertex[Dimension-1] ";
    ofs << "element[0] (...) element[Dimension-1] \n";

    ofs << Dimension << " ";
    ofs << _min.transpose() << " ";
    ofs << _max.transpose() << " ";
    ofs << (*_cell_length).transpose() << "\n";

    for (auto cell : _grid)
    {
      if (func(cell))
      {
        ofs << cell->vertex(0).transpose() << " ";
        ofs << cell->element() << "\n";
      }
    }
    ofs.close();
  }

  Iterator begin()
  {
    return _grid.begin();
  }

  Iterator end()
  {
    return _grid.end();
  }

protected:
  const Coordinate _min;
  const Coordinate _max;
  const CoordinatePtr _cell_length;
  const EigenArray _cell_length_inv;
  const Eigen::Vector<int, Dimension> _total_cells;
  const Eigen::Vector<double, Dimension> _2power;
  // const double _cell_length;

  std::vector<CellPtr> _grid;
};
}  // namespace utilities
}  // namespace prx
