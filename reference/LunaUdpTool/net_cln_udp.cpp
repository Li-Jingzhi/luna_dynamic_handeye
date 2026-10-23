#include "net_cln_udp.h"

CNetClnUdp::CNetClnUdp(const char* ip, unsigned short port)
    : sock_(INVALID_SOCKET), valid_(0) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return;
    }
    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ == INVALID_SOCKET) {
        WSACleanup();
        return;
    }
    // Zero out sockaddr_in manually (no memset from stdlib)
    char* p = (char*)&addr_;
    for (int i = 0; i < (int)sizeof(addr_); i++) p[i] = 0;

    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port);
    addr_.sin_addr.s_addr = inet_addr(ip);
    valid_ = 1;
}

CNetClnUdp::~CNetClnUdp() {
    if (sock_ != INVALID_SOCKET) {
        closesocket(sock_);
    }
    WSACleanup();
}

int CNetClnUdp::sendsvr(void* p, int len) {
    if (!valid_) return -1;
    return sendto(sock_, (const char*)p, len, 0,
                  (struct sockaddr*)&addr_, sizeof(addr_));
}

int CNetClnUdp::is_valid() const {
    return valid_;
}