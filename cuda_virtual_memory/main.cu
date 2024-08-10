#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <map>
#include <optional>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(CUresult err, const char* const func, const char* const file, const int line) {
    if (err != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(err, &errorString);
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err)
                  << " \"" << errorString << "\" " << func << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

/***********warpper for cuda api***********/
bool virtualAddressSupported(int device) {
    int virtualAddressing = 0;
    CHECK_CUDA_ERROR(cuDeviceGetAttribute(&virtualAddressing, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
    return virtualAddressing != 0;
}

CUdeviceptr allocateVirtualAddress(size_t size, size_t alignment = 0) {
    // Reserve a virtual address range with the specified size and alignment
    // When alignment is set to 0, the default alignment is used
    CUdeviceptr d_ptr;
    CHECK_CUDA_ERROR(cuMemAddressReserve(&d_ptr, size, alignment, 0, 0));
    return d_ptr;
}

void freeVirtualAddress(CUdeviceptr d_ptr, size_t size) {
    // Free the virtual address range
    CHECK_CUDA_ERROR(cuMemAddressFree(d_ptr, size));
}

size_t getMemoryGranularity(int device) {
    // Get the memory granularity of the device
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t granularity;
    CHECK_CUDA_ERROR(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return granularity;
}

CUmemGenericAllocationHandle allocatePhysicalMemory(size_t size, int device) {
    // Allocate physical memory on the specified device
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    
    size_t granularity = getMemoryGranularity(device);
    assert(size % granularity == 0);
    CHECK_CUDA_ERROR(cuMemCreate(&handle, size, &prop, 0));
    return handle;
}

void freePhysicalMemory(CUmemGenericAllocationHandle handle) {
    // Free the physical memory
    CHECK_CUDA_ERROR(cuMemRelease(handle));
}

//map physical memory to the allocated virtual address
void mapPA2VA(CUdeviceptr d_ptr, size_t size, CUmemGenericAllocationHandle handle, int device, int offset) {
    // Map physical memory to the reserved virtual address
    // virtual_memory[d_ptr, d_ptr + size) = physical_memory[offset, offset + size)
    // The handle is the handle to the physical allocation
    // The device is the device where the physical allocation resides
    CHECK_CUDA_ERROR(cuMemMap(d_ptr, size, offset, handle, 0));

    // Set access flags for the mapped memory
    CUmemAccessDesc accessDesc;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_ERROR(cuMemSetAccess(d_ptr, size, &accessDesc, 1));
}

void unmapPA2VA(CUdeviceptr d_ptr, size_t size) {
    // Unmap the physical memory
    CHECK_CUDA_ERROR(cuMemUnmap(d_ptr, size));
}
/***********warpper for cuda api***********/

int main() {
            CUcontext cuContext;
    CUdeviceptr d_ptr;
    int device = 0;
    size_t size = (1<<24) * sizeof(int); 

    // Initialize CUDA driver API
    CHECK_CUDA_ERROR(cuInit(0));
    CUdevice cuDevice;
    CHECK_CUDA_ERROR(cuDeviceGet(&cuDevice, device));
    CHECK_CUDA_ERROR(cuCtxCreate(&cuContext, 0, cuDevice));

    // Check if the device supports virtual address management
    if (!virtualAddressSupported(device)) {
        std::cerr << "Virtual address management is not supported on device " << device << std::endl;
        return;
    }
    // Reserve a virtual address range
    ////the minimum size of the virtual address range is 2MB, set alignment to 0 means the default alignment
    d_ptr = allocateVirtualAddress(size, 0);


    //request the allocation granularity of the device
    CUmemGenericAllocationHandle handle = allocatePhysicalMemory(size, device);

    // Map the physical memory to the reserved virtual address
    mapPA2VA(d_ptr, size, handle, device, 0);

    // Use the mapped memory
    int num_elements = size / sizeof(int);
    int* h_ptr = new int[num_elements];
    for (size_t i = 0; i < num_elements; ++i) {
        h_ptr[i] = static_cast<int>(i);
    }
    CHECK_CUDA_ERROR(cuMemcpyHtoD(d_ptr, h_ptr, size));

    // Print some values from the mapped memory
    int* h_ptr_out = new int[num_elements];
    CHECK_CUDA_ERROR(cuMemcpyDtoH(h_ptr_out, d_ptr, size));
    bool success = true;
    for (size_t i = 0; i < num_elements; ++i) {
        if (h_ptr_out[i] != h_ptr[i]) {
            std::cerr << "Mismatch at index " << i << ": " << h_ptr_out[i] << " != " << h_ptr[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "All values match" << std::endl;
    }

    // Unmap and release resources
    unmapPA2VA(d_ptr, size);
    freePhysicalMemory(handle);
    freeVirtualAddress(d_ptr, size);

    // Clean up
    delete[] h_ptr;
    delete[] h_ptr_out;
    CHECK_CUDA_ERROR(cuCtxDestroy(cuContext));
}