#pragma once

namespace SNICIT_SDGC {
    class GpuRunConfig {
    public:
        int block_num;
        int thread_num;
        int shared_memory_size;
    };

};