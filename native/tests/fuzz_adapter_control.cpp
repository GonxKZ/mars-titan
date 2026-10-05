#include "mars_titan/adapter_control.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    try {
        const std::string_view input(reinterpret_cast<const char*>(data), size);
        const auto state = mars_titan::controls::deserialize_adapter(input);
        const auto serialized = mars_titan::controls::serialize_adapter(state);
        const auto restored = mars_titan::controls::deserialize_adapter(serialized);
        if (mars_titan::controls::serialize_adapter(restored) != serialized)
            __builtin_trap();
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
