#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

struct SharedMemoryHandle {
    std::string triton_shm_name_;
    std::string shm_key_;
    cudaIpcMemHandle_t cuda_shm_handle_;
    int device_id_;
    void* base_addr_;
    int shm_fd_;
    size_t offset_;
    size_t byte_size_;
};
extern "C" {
    int CudaSharedMemoryRegionSet(
        void* cuda_shm_handle, size_t offset, size_t byte_size, const void* data)
    {
        // remember previous device and set to new device
        int previous_device;
        cudaGetDevice(&previous_device);
        cudaError_t err = cudaSetDevice(
            reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->device_id_);
        if (err != cudaSuccess) {
            cudaSetDevice(previous_device);
            return -1;
        }

        // Copy data into cuda shared memory
        void* base_addr =
            reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->base_addr_;
        std::cout <<"base_addr is :"<< (unsigned long long) base_addr << std::endl;
        err = cudaMemcpy(
            reinterpret_cast<uint8_t*>(base_addr) + offset, data, byte_size,
            cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaSetDevice(previous_device);
            return -3;
        }

        // Set device to previous GPU
        cudaSetDevice(previous_device);

        return 0;
    }

    int CudaSharedMemoryRegionSet_from_decoder(void* cuda_shm_handle, void* ptr_vpf,size_t offset, size_t byte_size) {
        // remember previous device and set to new device
        int previous_device;
        cudaGetDevice(&previous_device);
        cudaError_t err = cudaSetDevice(
            reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->device_id_);
        if (err != cudaSuccess) {
            cudaSetDevice(previous_device);
            return -1;
        }

        // Copy data into cuda shared memory
        void* base_addr = reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->base_addr_;
        auto res = cudaMemcpy(reinterpret_cast<uint8_t*>(base_addr) + offset, ptr_vpf, byte_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaSetDevice(previous_device);
            return -3;
        }

        // Set device to previous GPU
        cudaSetDevice(previous_device);

        return 0;
    }
}