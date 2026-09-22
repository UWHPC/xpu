#include <xpu/launch.hpp>
#include "../support/check.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <vector>

template <bool move_only = false, std::size_t dims>
auto check_range(const xpu::range<dims>& range) -> bool {
  // Serial Cartesian-product reference, independent of linear-index decoding.
  auto expected{std::vector<std::array<std::size_t, dims>>{}};
  auto index{range.begin};
  auto collect{[&](auto&& self, std::size_t dimension) -> void {
    auto coordinate{range.begin[dimension]};
    while (coordinate < range.end[dimension]) {
      index[dimension] = coordinate;
      if (dimension + 1uz == dims) {
        expected.push_back(index);
      } else {
        self(self, dimension + 1uz);
      }
      if (range.end[dimension] - coordinate <= range.step[dimension]) { break; }
      coordinate += range.step[dimension];
    }
  }};
  collect(collect, 0uz);

  auto visits{std::vector<unsigned int>(expected.size())};
  std::atomic<bool> invalid{false};
  auto visit{[&](const auto& coordinate, unsigned int value) {
    const auto found{std::find(expected.begin(), expected.end(), coordinate)};
    if (value != 7u || found == expected.end()) {
      invalid.store(true, std::memory_order_relaxed);
      return;
    }
    const auto slot{static_cast<std::size_t>(found - expected.begin())};
    std::atomic_ref<unsigned int>{visits[slot]}.fetch_add(1u, std::memory_order_relaxed);
  }};

  if constexpr (move_only) {
    auto callback{[owner{std::make_unique<unsigned int>(7u)}, &visit](const auto& coordinate) {
      visit(coordinate, *owner);
    }};
    xpu::parallel_for(range, callback);
  } else {
    xpu::parallel_for(range, [&](const auto& coordinate) { visit(coordinate, 7u); });
  }

  if (invalid.load(std::memory_order_relaxed)) { return false; }
  for (const auto count : visits) {
    if (count != 1u) { return false; }
  }
  return true;
}

auto main() -> int {
  if (!check_range<true>(xpu::range<1uz>{{0uz}, {8uz}, {1uz}})
      || !check_range<true>(xpu::range<1uz>{{1uz}, {8uz}, {2uz}})
      || !check_range<true>(xpu::range<2uz>{{0uz, 0uz}, {8uz, 8uz}, {1uz, 1uz}})
      || !check_range<true>(xpu::range<2uz>{{1uz, 1uz}, {8uz, 8uz}, {2uz, 2uz}})) {
    return test::fail("parallel_for failed with a move-only callback");
  }

  if (!check_range(xpu::range<1uz>{{0uz}, {257uz}, {1uz}})
      || !check_range(xpu::range<2uz>{{0uz, 0uz}, {17uz, 31uz}, {1uz, 1uz}})
      || !check_range(xpu::range<3uz>{{0uz, 0uz, 0uz}, {3uz, 5uz, 17uz}, {1uz, 1uz, 1uz}})
      || !check_range(xpu::range<4uz>{{0uz, 0uz, 0uz, 0uz}, {2uz, 3uz, 5uz, 7uz}, {1uz, 1uz, 1uz, 1uz}})) {
    return test::fail("parallel_for unit-step coordinates or visit counts differ");
  }

  if (!check_range(xpu::range<1uz>{{3uz}, {1003uz}, {7uz}})
      || !check_range(xpu::range<2uz>{{0uz, 0uz}, {17uz, 31uz}, {1uz, 3uz}})
      || !check_range(xpu::range<2uz>{{0uz, 2uz}, {17uz, 31uz}, {1uz, 1uz}})
      || !check_range(xpu::range<2uz>{{2uz, 5uz}, {35uz, 1003uz}, {3uz, 7uz}})
      || !check_range(xpu::range<3uz>{{1uz, 2uz, 3uz}, {9uz, 13uz, 25uz}, {2uz, 3uz, 4uz}})
      || !check_range(xpu::range<4uz>{{1uz, 2uz, 3uz, 4uz}, {5uz, 7uz, 9uz, 11uz}, {2uz, 2uz, 3uz, 3uz}})) {
    return test::fail("parallel_for coordinates or visit counts differ");
  }

  constexpr auto max{std::numeric_limits<std::size_t>::max()};
  if (!check_range(xpu::range<1uz>{{max - 8uz}, {max}, {3uz}})
      || !check_range(xpu::range<2uz>{{max - 8uz, max - 8uz}, {max, max}, {3uz, 3uz}})) {
    return test::fail("parallel_for failed near SIZE_MAX");
  }

  const auto unexpected{[](const auto&) { std::abort(); }};
  xpu::parallel_for(xpu::range<1uz>{{0uz}, {10uz}, {0uz}}, unexpected);
  xpu::parallel_for(xpu::range<2uz>{{0uz, 5uz}, {10uz, 5uz}, {1uz, 1uz}}, unexpected);
  xpu::parallel_for(xpu::range<2uz>{{0uz, 6uz}, {10uz, 5uz}, {1uz, 1uz}}, unexpected);

  // An empty later dimension suppresses overflow in earlier dimensions.
  xpu::parallel_for(xpu::range<3uz>{{0uz, 0uz, 0uz}, {max, max, 0uz}, {1uz, 1uz, 1uz}}, unexpected);
  return 0;
}
