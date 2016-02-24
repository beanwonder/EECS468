#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto( /*Define your own function parameters*/ );

/* Include below the function headers of any other functions that you implement */
uint32_t* allocate_device_histogram_bins(size_t height, size_t width);
uint8_t* allocate_device_bins(size_t height, size_t width);
void copy_to_device_histogram_bins(uint32_t *device, const uint32_t *host, size_t h, size_t w);
void copy_from_device_bins(uint8_t *host, const uint8_t *device, size_t h, size_t w);

#endif
