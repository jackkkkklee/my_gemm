//
// Created by 76919 on 2023/3/3.
//

#include "tool.h"

inline int ConvertSMVer2Cores(int major, int minor) {
    // CUDA Capability 2.x devices
    if (major == 2) {
        switch (minor) {
            case 0: return 32;  // Fermi Generation (GF100, GF110)
            case 1: return 48;  // Fermi Generation (GF104, GF106 GF108)
        }
        // CUDA Capability 3.x devices
    } else if (major == 3) {
        switch (minor) {
            case 0: return 192; // Kepler Generation (GK10x)
            case 5: return 192; // Kepler Generation (GK11x)
        }
        // CUDA Capability 5.x devices
    } else if (major == 5) {
        switch (minor) {
            case 0: return 128; // Maxwell Generation (GM10x)
            case 2: return 128; // Maxwell Generation (GM20x)
        }
        // CUDA Capability 6.x devices
    } else if (major == 6) {
        switch (minor) {
            case 0: return 64;  // Pascal Generation (GP100, GP102, GP104, GP106, GP107)
            case 1: return 128; // Volta Generation (GV100, GV10B)
            case 2: return 128; // Turing Generation (TU102, TU104, TU106)
        }
        // CUDA Capability 7.x devices
    } else if (major == 7) {
        switch (minor) {
            case 0: return 64;  // Ampere Generation (GA10x)
        }
    }
    return 0;
}
