/* CPU-only ABI oracle: includes the header, never links or calls NVML. */
#include <stddef.h>
#include <stdio.h>
#include <nvml.h>

#define SIZE(type) printf("\"" #type "\":%zu,", sizeof(type))
#define OFFSET(type, field) printf("\"" #type "." #field "\":%zu,", offsetof(type, field))

int main(void) {
    printf("{");
    SIZE(nvmlPciInfo_t);
    OFFSET(nvmlPciInfo_t, busIdLegacy);
    OFFSET(nvmlPciInfo_t, domain);
    OFFSET(nvmlPciInfo_t, bus);
    OFFSET(nvmlPciInfo_t, device);
    OFFSET(nvmlPciInfo_t, pciDeviceId);
    OFFSET(nvmlPciInfo_t, pciSubSystemId);
    OFFSET(nvmlPciInfo_t, busId);
    SIZE(nvmlMemory_v2_t);
    OFFSET(nvmlMemory_v2_t, version);
    OFFSET(nvmlMemory_v2_t, total);
    OFFSET(nvmlMemory_v2_t, reserved);
    OFFSET(nvmlMemory_v2_t, free);
    OFFSET(nvmlMemory_v2_t, used);
    SIZE(nvmlUtilization_t);
    OFFSET(nvmlUtilization_t, gpu);
    OFFSET(nvmlUtilization_t, memory);
    SIZE(nvmlValue_t);
    OFFSET(nvmlValue_t, dVal);
    OFFSET(nvmlValue_t, siVal);
    OFFSET(nvmlValue_t, uiVal);
    OFFSET(nvmlValue_t, ulVal);
    OFFSET(nvmlValue_t, ullVal);
    OFFSET(nvmlValue_t, sllVal);
    SIZE(nvmlFieldValue_t);
    OFFSET(nvmlFieldValue_t, fieldId);
    OFFSET(nvmlFieldValue_t, scopeId);
    OFFSET(nvmlFieldValue_t, timestamp);
    OFFSET(nvmlFieldValue_t, latencyUsec);
    OFFSET(nvmlFieldValue_t, valueType);
    OFFSET(nvmlFieldValue_t, nvmlReturn);
    OFFSET(nvmlFieldValue_t, value);
    printf("\"memory_version\":%u,\"integers\":[%d,%d,%d,%d,%d]}\n", nvmlMemory_v2,
           NVML_VALUE_TYPE_UNSIGNED_INT, NVML_VALUE_TYPE_UNSIGNED_LONG,
           NVML_VALUE_TYPE_UNSIGNED_LONG_LONG, NVML_VALUE_TYPE_SIGNED_LONG_LONG,
           NVML_VALUE_TYPE_SIGNED_INT);
    return 0;
}
