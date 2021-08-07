#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class SVM {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float kernels[75] = { 0 };
                        float decisions[3] = { 0 };
                        int votes[3] = { 0 };
                        kernels[0] = compute_kernel(x,   5.66  , -8.29  , 0.16 );
                        kernels[1] = compute_kernel(x,   2.57  , -8.59  , -4.25 );
                        kernels[2] = compute_kernel(x,   5.45  , -8.18  , 0.19 );
                        kernels[3] = compute_kernel(x,   5.33  , -8.22  , -0.6 );
                        kernels[4] = compute_kernel(x,   1.27  , -8.31  , -3.48 );
                        kernels[5] = compute_kernel(x,   5.49  , -8.13  , 0.05 );
                        kernels[6] = compute_kernel(x,   1.63  , -6.54  , 2.7 );
                        kernels[7] = compute_kernel(x,   1.55  , -6.42  , 2.52 );
                        kernels[8] = compute_kernel(x,   5.42  , -8.1  , -0.64 );
                        kernels[9] = compute_kernel(x,   5.83  , -8.68  , 0.4 );
                        kernels[10] = compute_kernel(x,   1.57  , -6.39  , 2.62 );
                        kernels[11] = compute_kernel(x,   4.96  , -8.08  , -1.04 );
                        kernels[12] = compute_kernel(x,   5.43  , -8.28  , 0.45 );
                        kernels[13] = compute_kernel(x,   1.34  , -8.3  , -3.51 );
                        kernels[14] = compute_kernel(x,   1.45  , -6.57  , 2.54 );
                        kernels[15] = compute_kernel(x,   5.48  , -8.1  , -0.67 );
                        kernels[16] = compute_kernel(x,   4.87  , -7.98  , -1.2 );
                        kernels[17] = compute_kernel(x,   5.43  , -8.06  , 0.35 );
                        kernels[18] = compute_kernel(x,   5.41  , -8.07  , -0.67 );
                        kernels[19] = compute_kernel(x,   5.6  , -8.18  , 0.04 );
                        kernels[20] = compute_kernel(x,   5.5  , -7.97  , 0.15 );
                        kernels[21] = compute_kernel(x,   4.81  , -7.91  , -1.42 );
                        kernels[22] = compute_kernel(x,   5.58  , -8.02  , 0.03 );
                        kernels[23] = compute_kernel(x,   3.81  , 8.9  , 1.31 );
                        kernels[24] = compute_kernel(x,   4.28  , 7.88  , -4.31 );
                        kernels[25] = compute_kernel(x,   6.19  , 6.96  , -4.58 );
                        kernels[26] = compute_kernel(x,   -2.07  , 5.74  , -7.33 );
                        kernels[27] = compute_kernel(x,   4.53  , 8.27  , -2.77 );
                        kernels[28] = compute_kernel(x,   4.85  , 7.92  , -2.29 );
                        kernels[29] = compute_kernel(x,   6.93  , 8.01  , -1.47 );
                        kernels[30] = compute_kernel(x,   4.2  , 7.81  , -4.69 );
                        kernels[31] = compute_kernel(x,   3.55  , 6.95  , -7.77 );
                        kernels[32] = compute_kernel(x,   0.61  , 5.38  , -6.82 );
                        kernels[33] = compute_kernel(x,   -1.19  , 8.24  , 1.08 );
                        kernels[34] = compute_kernel(x,   4.38  , 8.26  , -3.34 );
                        kernels[35] = compute_kernel(x,   -2.55  , 5.64  , -7.35 );
                        kernels[36] = compute_kernel(x,   4.97  , 7.98  , -1.66 );
                        kernels[37] = compute_kernel(x,   4.67  , 8.15  , 0.74 );
                        kernels[38] = compute_kernel(x,   4.41  , 8.35  , -2.97 );
                        kernels[39] = compute_kernel(x,   4.71  , 8.3  , -1.56 );
                        kernels[40] = compute_kernel(x,   4.66  , 8.15  , -2.6 );
                        kernels[41] = compute_kernel(x,   5.06  , 7.84  , -1.79 );
                        kernels[42] = compute_kernel(x,   4.19  , 7.77  , -5.04 );
                        kernels[43] = compute_kernel(x,   9.01  , 1.67  , -0.89 );
                        kernels[44] = compute_kernel(x,   10.29  , -1.92  , -1.11 );
                        kernels[45] = compute_kernel(x,   10.53  , -2.41  , 0.78 );
                        kernels[46] = compute_kernel(x,   8.37  , 2.54  , -0.7 );
                        kernels[47] = compute_kernel(x,   9.49  , 1.7  , -2.94 );
                        kernels[48] = compute_kernel(x,   8.78  , 1.93  , -0.52 );
                        kernels[49] = compute_kernel(x,   9.65  , 2.0  , 1.12 );
                        kernels[50] = compute_kernel(x,   9.18  , -1.13  , 0.13 );
                        kernels[51] = compute_kernel(x,   9.99  , 1.91  , -1.47 );
                        kernels[52] = compute_kernel(x,   9.56  , -1.07  , -0.98 );
                        kernels[53] = compute_kernel(x,   9.55  , -0.93  , -0.48 );
                        kernels[54] = compute_kernel(x,   10.41  , -2.01  , 0.56 );
                        kernels[55] = compute_kernel(x,   9.89  , -1.34  , -1.02 );
                        kernels[56] = compute_kernel(x,   9.39  , -0.8  , -0.8 );
                        kernels[57] = compute_kernel(x,   9.45  , 0.91  , -3.44 );
                        kernels[58] = compute_kernel(x,   9.65  , -1.02  , -1.18 );
                        kernels[59] = compute_kernel(x,   10.58  , -1.86  , 0.83 );
                        kernels[60] = compute_kernel(x,   9.56  , -1.02  , -0.81 );
                        kernels[61] = compute_kernel(x,   9.46  , 1.55  , -0.65 );
                        kernels[62] = compute_kernel(x,   10.35  , -1.76  , -1.32 );
                        kernels[63] = compute_kernel(x,   9.81  , 1.73  , -1.69 );
                        kernels[64] = compute_kernel(x,   9.72  , 1.19  , -3.38 );
                        kernels[65] = compute_kernel(x,   10.11  , 1.86  , -1.11 );
                        kernels[66] = compute_kernel(x,   10.45  , -1.73  , -0.02 );
                        kernels[67] = compute_kernel(x,   9.58  , 1.05  , -3.54 );
                        kernels[68] = compute_kernel(x,   9.91  , 1.88  , -2.17 );
                        kernels[69] = compute_kernel(x,   10.37  , -1.56  , -0.79 );
                        kernels[70] = compute_kernel(x,   8.59  , 2.69  , -1.13 );
                        kernels[71] = compute_kernel(x,   8.41  , 3.11  , -1.09 );
                        kernels[72] = compute_kernel(x,   9.56  , -1.03  , -1.13 );
                        kernels[73] = compute_kernel(x,   10.27  , -1.89  , 0.32 );
                        kernels[74] = compute_kernel(x,   9.49  , 1.52  , -3.15 );
                        decisions[0] = -0.075562822348
                        + kernels[1] * 0.188409689959
                        + kernels[4]
                        + kernels[6] * 0.186361378642
                        + kernels[7]
                        + kernels[10]
                        + kernels[13]
                        + kernels[14]
                        + kernels[25] * -0.712497577233
                        + kernels[26] * -0.668642492001
                        - kernels[32]
                        + kernels[33] * -0.993630999367
                        - kernels[35]
                        - kernels[37]
                        ;
                        decisions[1] = -0.005657047033
                        + kernels[0]
                        + kernels[2]
                        + kernels[3]
                        + kernels[5]
                        + kernels[8]
                        + kernels[9] * 0.094722489834
                        + kernels[11]
                        + kernels[12]
                        + kernels[15]
                        + kernels[16]
                        + kernels[17]
                        + kernels[18]
                        + kernels[19]
                        + kernels[20]
                        + kernels[21]
                        + kernels[22]
                        - kernels[44]
                        - kernels[45]
                        - kernels[50]
                        - kernels[52]
                        - kernels[53]
                        - kernels[54]
                        - kernels[55]
                        + kernels[56] * -0.094722489834
                        - kernels[58]
                        - kernels[59]
                        - kernels[60]
                        - kernels[62]
                        - kernels[66]
                        - kernels[69]
                        - kernels[72]
                        - kernels[73]
                        ;
                        decisions[2] = 0.02919868758
                        + kernels[23] * 0.399427400564
                        + kernels[24]
                        + kernels[25]
                        + kernels[27]
                        + kernels[28]
                        + kernels[29]
                        + kernels[30]
                        + kernels[31] * 0.824961887878
                        + kernels[34]
                        + kernels[36]
                        + kernels[37]
                        + kernels[38]
                        + kernels[39]
                        + kernels[40]
                        + kernels[41]
                        + kernels[42]
                        - kernels[43]
                        - kernels[46]
                        - kernels[47]
                        - kernels[48]
                        - kernels[49]
                        - kernels[51]
                        - kernels[57]
                        - kernels[61]
                        - kernels[63]
                        - kernels[64]
                        + kernels[65] * -0.224389288442
                        - kernels[67]
                        - kernels[68]
                        - kernels[70]
                        - kernels[71]
                        - kernels[74]
                        ;
                        votes[decisions[0] > 0 ? 0 : 1] += 1;
                        votes[decisions[1] > 0 ? 0 : 2] += 1;
                        votes[decisions[2] > 0 ? 1 : 2] += 1;
                        int val = votes[0];
                        int idx = 0;

                        for (int i = 1; i < 3; i++) {
                            if (votes[i] > val) {
                                val = votes[i];
                                idx = i;
                            }
                        }

                        return idx;
                    }

                    /**
                    * Predict readable class name
                    */
                    const char* predictLabel(float *x) {
                        return idxToLabel(predict(x));
                    }

                    /**
                    * Convert class idx to readable name
                    */
                    const char* idxToLabel(uint8_t classIdx) {
                        switch (classIdx) {
                            case 0:
                            return "UP";
                            case 1:
                            return "DOWN";
                            case 2:
                            return "HORIZONTAL";
                            default:
                            return "Houston we have a problem";
                        }
                    }

                protected:
                    /**
                    * Compute kernel between feature vector and support vector.
                    * Kernel type: rbf
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, 3);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 3; i++) {
                            kernel += pow(x[i] - va_arg(w, double), 2);
                        }

                        return exp(-0.001 * kernel);
                    }
                };
            }
        }
    }