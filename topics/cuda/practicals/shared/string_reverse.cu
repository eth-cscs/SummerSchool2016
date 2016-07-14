#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.h"

// TODO : implement a kernel that reverses a string of length n in place
// void reverse_string(char* str, int n)

int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_host<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << std::endl;

    // allocate memory on device and copy string into it
    auto string_d = malloc_device<char>(n);
    copy_to_device<char>(string, string_d, n);

    // TODO : call the string reverse function

    // copy reversed string back to host and print
    copy_to_host<char>(string_d, string, n);
    std::cout << "reversed string:\n" << string << std::endl;

    // free memory used in 
    free(string);
    cudaFree(string);

    return 0;
}

