//
//  main.cpp
//  logicle_v2
//
//  Created by Jonathan Henninger on 1/20/18.
//  Copyright © 2018 Jonathan Henninger. All rights reserved.
//

#include <iostream>
#include <vector>
#include <sstream>
#include <iterator>
#include "logicle.h"
using namespace std;

int main(int argc, const char * argv[]) {
   /* good defaults */
    double T = atof(argv[1]);
    double W = atof(argv[2]);
    double M = atof(argv[3]);
    double A = atof(argv[4]);
//    double T = 262144; /* 262144 */
//    double W = 0.5; /* 0.5 */
//    double M = 4.5; /* 4.5 */
//    double A = 0.0;
    Logicle l = Logicle(T, W, M, A);
    
    // vector<double> result;
    
    for (int i = 5; i < argc; i++) {
        double value = atof(argv[i]);
        double result = l.scale(value);
        cout << result << endl;
        // result[i-5] = l.scale(value);
        }
    //stringstream output;
   // copy(result.begin(), result.end(), ostream_iterator<double>(output, " "));
    
   // std::cout << output.str() << endl;

    return 0;
}
