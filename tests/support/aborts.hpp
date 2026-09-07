#pragma once

#if defined(__unix__)

#include "check.hpp"

#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace test {

template <typename Function>
auto check_abort(Function function) -> int {
  const auto child{fork()};
  const auto child_started{child >= 0};

  if (const auto failed{!child_started}; failed) {
    const auto failure{test::fail("could not start failure test")};

    return failure;
  }

  const auto in_child{child == 0};

  if (in_child) {
    const auto core_limit = rlimit{0, 0};
    setrlimit(RLIMIT_CORE, &core_limit);
    function();
    _exit(0);
  }

  auto status{0};
  const auto child_collected{waitpid(child, &status, 0) == child};

  if (const auto failed{!child_collected}; failed) {
    const auto failure{test::fail("could not collect failure test")};

    return failure;
  }

  const auto aborted{WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT};

  if (const auto failed{!aborted}; failed) {
    const auto failure{test::fail("invalid arithmetic did not abort")};

    return failure;
  }

  const auto success{0};

  return success;
}

} // namespace test

#endif
