#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <functional>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;

class MPI_Wrapper
{
private:
    int num_procs, p_id;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Status status;

    inline static constexpr int DEFAULT_TAG = 0;
public:
    MPI_Wrapper();
    int get_procs_count() const;
    void execute_on(int id, function<void(MPI_Wrapper*, int)> f);
    void execute_on_others(int id, function<void(MPI_Wrapper*, int)> f);
    void send_int(int to_p, int x);
    int recv_int(int from_p);
    void send_ints(int to_p, const int* buffer, int count);
    void recv_ints(int from_p, int* buffer, int count);
    void broadcast(int root, int* buffer, int count);
    void execute_this(function<void(MPI_Wrapper*, int)> f);
    void scatter_ints(int root, int* whole, int* part, int count);
    void gather_ints(int root, int* whole, int* part, int count);

    ~MPI_Wrapper();
};

template<class Fn, class... Types>
void measure(Fn&& f, Types&&... args)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    f(args...);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf(">>Measured time = %f\n", elapsed_time_ms);    
}

int conv(int* K, int* prvL, int* crtL, int* nxtL, int j, int M)
{
    return K[0] * prvL[max(j - 1, 0)] + K[1] * prvL[j] + K[2] * prvL[min(j + 1, M - 1)]
         + K[3] * crtL[max(j - 1, 0)] + K[4] * crtL[j] + K[5] * crtL[min(j + 1, M - 1)]
         + K[6] * nxtL[max(j - 1, 0)] + K[7] * nxtL[j] + K[8] * nxtL[min(j + 1, M - 1)];
}

void generate_ints();

int main_v2(int argc, char** argv)
{
    MPI_Wrapper mpi;
    int N = atoi(argv[1]), M = N;

    int K[9];
    int* A = new int[N * M];

    mpi.execute_on(0, [&](MPI_Wrapper* ctx, int id)
        {
            FILE* f = fopen("date.bin", "r");
            fread(K, sizeof(int), 9, f);
            fread(A, sizeof(int), N * M, f);
            fclose(f);
        });

    auto t_start = std::chrono::high_resolution_clock::now();   

    mpi.broadcast(0, K, 9);

    int p_lines_count = N / mpi.get_procs_count();


    int* mat = new int[p_lines_count * M];

    mpi.scatter_ints(0, A, mat, p_lines_count * M);    

    int* prevLine = new int[M];
    int* crtLine = new int[M];
    int* lastLine = new int[M];

    mpi.execute_on(0, [&](MPI_Wrapper* ctx, int id)
        {
            int s = 0, e = min(p_lines_count, M - 1);
            memcpy(prevLine, &A[s * M], M * sizeof(int));
            memcpy(lastLine, &A[e * M], M * sizeof(int));            

            for (int p = 1; p < ctx->get_procs_count(); p++)
            {
                s = p * p_lines_count - 1;
                e = min((p + 1) * p_lines_count, M - 1);                

                ctx->send_ints(p, &A[s * M], M);
                ctx->send_ints(p, &A[e * M], M);
            }
        });

    mpi.execute_on_others(0, [&](MPI_Wrapper* ctx, int id)
        {
            ctx->recv_ints(0, prevLine, M);
            ctx->recv_ints(0, lastLine, M);
        });


    for (int i = 0; i < p_lines_count; i++)
    {
        memcpy(crtLine, &mat[i * M], M * sizeof(int));
        int* nextLine = i < p_lines_count - 1 ? &mat[(i + 1) * M] : lastLine;
        for (int j = 0; j < M; j++)
            mat[i * M + j] = conv(K, prevLine, crtLine, nextLine, j, M);
        memcpy(prevLine, crtLine, M * sizeof(int));
    }    

    mpi.gather_ints(0, A, mat, p_lines_count * M);        
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();    


    mpi.execute_on(0, [&](MPI_Wrapper* ctx, int id)
        {
            printf(">>Measured time = %f\n", elapsed_time_ms);

            ofstream g("result.txt");
            g << N << " " << M << "\n";
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                    g << A[i * M + j] << " ";
                g << "\n";
            }
            g.close();
        });

    return 0;
}

int main_v1(int argc, char** argv) 
{            
    MPI_Wrapper mpi;   
    int N = atoi(argv[1]), M = N;

    mpi.execute_on(0, [&](MPI_Wrapper* ctx, int id)
        {            
            int K[9];
            int* A = new int[N * M];

            measure([&]() 
                {
                    FILE* f = fopen("date.bin", "r");

                    fread(K, sizeof(int), 9, f);

                    for (int p = 1; p < ctx->get_procs_count(); p++)
                        ctx->send_ints(p, K, 9);

                    int chunk_lines_count = N / (ctx->get_procs_count() - 1);
                    printf("Reading %i lines at a time\n", chunk_lines_count);

                    int line_index = 0;

                    for (int p = 1; p < ctx->get_procs_count(); p++)
                    {
                        ctx->send_int(p, chunk_lines_count); // how many lines to read

                        fread(&A[line_index * M], sizeof(int), chunk_lines_count * M, f);

                        // first line owned by p is the next line in p-1, so send it to the previous thread
                        if (p > 1)
                            ctx->send_ints(p - 1, &A[line_index * M], M);

                        // send previous line to p                
                        ctx->send_ints(p, &A[max(line_index - 1, 0) * M], M);

                        // send its own block to p                
                        ctx->send_ints(p, &A[line_index * M], chunk_lines_count * M);

                        line_index += chunk_lines_count;
                    }
                    fclose(f);

                    // send the last line of the matrix as the next line of the last thread
                    ctx->send_ints(ctx->get_procs_count() - 1, &A[(N - 1) * M], M);

                    int* recvA = A;
                    for (int p = 1; p < ctx->get_procs_count(); p++)
                    {
                        ctx->recv_ints(p, recvA, chunk_lines_count * M);
                        recvA += chunk_lines_count * M;
                    }
                });

            ofstream g("result.txt");
            g << N << " " << M << "\n";
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                    g << A[i * M + j] << " ";
                g << "\n";
            }
            g.close();
            
            delete[] A;                       
        });

    mpi.execute_on_others(0, [&](MPI_Wrapper* ctx, int id)
        {
            int K[9];            
            ctx->recv_ints(0, K, 9);

            // each thread receives in order: prevLine, no. its own lines, its own submatrix, and lastLine

            int n = ctx->recv_int(0);

            int* prevLine = new int[M];
            int* mat = new int[n * M];
            int* lastLine = new int[M];

            ctx->recv_ints(0, prevLine, M);
            ctx->recv_ints(0, mat, n * M);
            ctx->recv_ints(0, lastLine, M);

            int* crtLine = new int[M];

            for (int i = 0; i < n; i++)
            {
                memcpy(crtLine, &mat[i * M], M * sizeof(int));
                int* nextLine = i < n - 1 ? &mat[(i + 1) * M] : lastLine;
                for (int j = 0; j < M; j++)
                    mat[i * M + j] = conv(K, prevLine, crtLine, nextLine, j, M);
                memcpy(prevLine, crtLine, M * sizeof(int));
            }

            ctx->send_ints(0, mat, n * M);            
        });

    return 0; 
}

int main(int argc, char** argv)
{
    if (argc >= 3 && strcmp(argv[2], "v2")==0)
        return main_v2(argc, argv);
    else return main_v1(argc, argv);
}

void generate_ints()
{
    FILE* f;
    fopen_s(&f, "date.bin", "w");
    int cnt = 1000009;
    int* buf = new int[cnt]();
    for (int i = 0; i < cnt; i++)
        buf[i] = rand() % 10;
    fwrite(buf, sizeof(int), cnt, f);
    delete[] buf;
    fclose(f);
}


MPI_Wrapper::MPI_Wrapper()
{
    status = MPI_Status{};
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
    MPI_Get_processor_name(processor_name, &name_len);
}

int MPI_Wrapper::get_procs_count() const { return num_procs; }

void MPI_Wrapper::execute_on(int id, function<void(MPI_Wrapper*, int)> f)
{
    if (p_id == id) f(this, p_id);
}

void MPI_Wrapper::execute_on_others(int id, function<void(MPI_Wrapper*, int)> f)
{
    if (p_id != id) f(this, p_id);
}

void MPI_Wrapper::execute_this(function<void(MPI_Wrapper*, int)> f)
{
    f(this, p_id);
}

void MPI_Wrapper::send_int(int to_p, int x)
{
    MPI_Send(&x, 1, MPI_INT, to_p, DEFAULT_TAG, MPI_COMM_WORLD);
}

int MPI_Wrapper::recv_int(int from_p)
{
    int x;
    MPI_Recv(&x, 1, MPI_INT, from_p, DEFAULT_TAG, MPI_COMM_WORLD, &status);
    return x;
}

void MPI_Wrapper::send_ints(int to_p, const int* buffer, int count)
{
    MPI_Send(buffer, count, MPI_INT, to_p, DEFAULT_TAG, MPI_COMM_WORLD);
}

void MPI_Wrapper::recv_ints(int from_p, int* buffer, int count)
{
    MPI_Recv(buffer, count, MPI_INT, from_p, DEFAULT_TAG, MPI_COMM_WORLD, &status);
}

void MPI_Wrapper::broadcast(int root, int* buffer, int count)
{
    MPI_Bcast(buffer, count, MPI_INT, root, MPI_COMM_WORLD);
}

void MPI_Wrapper::scatter_ints(int root, int* whole, int* part, int count)
{
    MPI_Scatter(whole, count, MPI_INT, part, count, MPI_INT, root, MPI_COMM_WORLD);
}

void MPI_Wrapper::gather_ints(int root, int* whole, int* part, int count)
{
    MPI_Gather(part, count, MPI_INT, whole, count, MPI_INT, root, MPI_COMM_WORLD);
}

MPI_Wrapper::~MPI_Wrapper()
{
    MPI_Finalize();
}