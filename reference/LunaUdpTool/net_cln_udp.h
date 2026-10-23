#pragma once
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

// Network UDP client - sends joint angle data to MuJoCo viewer.
// Does NOT use C++ standard library.
class CNetClnUdp {
public:
    CNetClnUdp(const char* ip, unsigned short port);
    ~CNetClnUdp();
    int sendsvr(void* p, int len);
    int is_valid() const;

private:
    SOCKET sock_;
    struct sockaddr_in addr_;
    int valid_;
};