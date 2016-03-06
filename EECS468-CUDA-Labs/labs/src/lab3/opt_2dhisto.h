#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t *result, uint32_t *input, int height, int width);

/* Include below the function headers of any other functions that you implement */

uint32_t *histogram_data_device(uint32_t** input, int height, int width);

void* allocate_device(size_t s);
void copy_to_device(void *device, void *host, size_t s);
void copy_from_device(void* host, void *device, size_t s);
void bin32_to_bin8(uint32_t *k32, uint8_t *k8);

void free_device(void* device);
// void copy_final(uint32_t *data, uint32_t  *histo_bin, uint32_t *histo);

// uint32_t ** setup_data(uint32_t ** input, int width, int height);

// uint8_t * setup_histo();

// uint32_t * setup(uint32_t * input, int width,int height);


// uint32_t * setuphisto();
#endif
