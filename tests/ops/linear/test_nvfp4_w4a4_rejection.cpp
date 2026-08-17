// Negative contract of the sm_90a build: every nvfp4 W4A4 route must fail with
// the Blackwell build error, and the portable A16 routes must stay admitted.
// The checks are host-side only; no CUDA device or context is required, so the
// binary runs wherever the 90a test binary can execute.
#ifdef NINFER_NVFP4_W4A4

#include <iostream>

int main() {
    std::cout << "SKIP: this build compiles the nvfp4 W4A4 kernels\n";
    return 77;
}

#else

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <string>

namespace {

template <class Capacity>
int expect_blackwell_error(const char* label, Capacity&& capacity) {
    try {
        (void)capacity();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("Blackwell") == std::string::npos) {
            std::cerr << label << ": unexpected rejection message: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << label << ": W4A4 route was not rejected\n";
    return 1;
}

} // namespace

int main() {
    const auto a4  = ninfer::ops::LinearPolicy::AllowA4;
    const auto a16 = ninfer::ops::LinearPolicy::A16Only;

    int failures = 0;
    failures += expect_blackwell_error("nvfp4 linear", [&] {
        return ninfer::ops::detail::nvfp4_linear_workspace_capacity_bytes(14336, 5120, a4, 16,
                                                                          1024);
    });
    failures += expect_blackwell_error("nvfp4 attn_input_proj", [&] {
        return ninfer::ops::detail::nvfp4_attn_input_workspace_capacity_bytes(a4, 16, 1024);
    });
    failures += expect_blackwell_error("nvfp4 gdn_input_proj", [&] {
        return ninfer::ops::detail::nvfp4_gdn_input_workspace_capacity_bytes(a4, 16, 1024);
    });
    failures += expect_blackwell_error("nvfp4 linear_add", [&] {
        return ninfer::ops::detail::nvfp4_linear_add_workspace_capacity_bytes(5120, 6144, a4, 16,
                                                                             1024);
    });
    failures += expect_blackwell_error("nvfp4 linear_swiglu", [&] {
        return ninfer::ops::detail::nvfp4_linear_swiglu_workspace_capacity_bytes(a4, 16, 1024);
    });

    // The portable A16 routes remain admitted and request no workspace.
    if (ninfer::ops::detail::nvfp4_linear_workspace_capacity_bytes(14336, 5120, a16, 16, 16) != 0 ||
        ninfer::ops::detail::nvfp4_linear_swiglu_workspace_capacity_bytes(a16, 1, 16) != 0) {
        std::cerr << "A16-only nvfp4 routes were not admitted\n";
        ++failures;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " nvfp4 W4A4 rejection (90a build)\n";
    return failures == 0 ? 0 : 1;
}

#endif
