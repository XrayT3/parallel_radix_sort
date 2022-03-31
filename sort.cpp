#include "sort.h"
#include <iostream>
#include <array>
#include <vector>
#include <atomic>
#include <mutex>
#include "omp.h"
#include <map>

using namespace std;

void save_vector(vector<string *> &vector_to_sort, int alphabet_size, map<size_t, 
                 vector< vector<string*>>> &thread_tables, vector<unsigned int> &pref_sum)
{
    size_t cnt_threads = thread_tables.size();
    
    #pragma omp parallel for
    for (int letter = 0; letter < alphabet_size; letter++) {
        size_t index = 0;
        for (size_t thread = 0; thread < cnt_threads; thread++) {
            if (thread_tables[thread].empty())
                continue;
            copy(thread_tables[thread][letter].begin(), thread_tables[thread][letter].end(), 
            vector_to_sort.begin() + pref_sum[letter] + index);
            index += thread_tables[thread][letter].size();
        }
    }
}


void radixStep(vector<string *> &vector_to_sort, int alphabet_size, const MappingFunction &mappingFunction, int pos)
{
    int c;                                          // char as array index for Radix sort   
    size_t size_vector = vector_to_sort.size();
    vector<unsigned int> pref_sum = {0, 0, 0, 0, 0, 0};
    map<size_t, vector< vector<string*>>> thread_tables;
    map<size_t, vector<unsigned int>> pref_sums;
    #pragma omp parallel
    {
        const unsigned int chunk_size = 1 + size_vector / omp_get_num_threads();
        const unsigned int thread_id = omp_get_thread_num();
        const unsigned int begin = thread_id * chunk_size;
        unsigned int end = (thread_id + 1) * chunk_size;
        if (end > size_vector) end = size_vector;

        vector<unsigned int> thread_pref_sum = {0, 0, 0, 0, 0, 0};
        vector<vector<string *>> thread_table(alphabet_size);
        for (size_t i = begin; i < end; ++i) {
            c = mappingFunction( vector_to_sort[i]->at(pos) ); 
            thread_pref_sum[c+1]++;
            thread_table[c].push_back(vector_to_sort[i]);
        }
        #pragma omp critical
        {
            pref_sums[thread_id] = move(thread_pref_sum);
            thread_tables[thread_id] = move(thread_table);
        }
    }
    //pref sum
    size_t cnt_threads = pref_sums.size();
    for(int i = 1; i < alphabet_size; i++){
        for (size_t thread = 0; thread < cnt_threads; thread++) {
            pref_sum[i] += pref_sums[thread][i];
        }
        pref_sum[i] += pref_sum[i-1];
    }
    save_vector(vector_to_sort, alphabet_size, thread_tables, pref_sum);
}


void radix_par(vector<string *> &vector_to_sort, const MappingFunction &mappingFunction,
               unsigned long alphabet_size, unsigned long string_lengths) {

    for( int pos = string_lengths - 1; pos >= 0; pos-- ) {
        radixStep( vector_to_sort, alphabet_size, mappingFunction, pos );       
    }

}
