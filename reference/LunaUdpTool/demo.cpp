#include "net_cln_udp.h"
#include <windows.h>
#include <math.h>

#define NUM_JOINTS 20
#define PI 3.14159265358979323846f

// Data packet: 20 floats (80 bytes)
//   [0:6]   = Leg_J1 ~ Leg_J6   (6 joints)
//   [6:13]  = Arm_R1 ~ Arm_R7   (7 joints, right arm)
//   [13:20] = Arm_L1 ~ Arm_L7   (7 joints, left arm)

int main() {
    CNetClnUdp client("127.0.0.1", 9999);
    if (!client.is_valid()) {
        return -1;
    }

    float angles[NUM_JOINTS];
    // Zero the angles array (no stdlib memset)
    for (int i = 0; i < NUM_JOINTS; i++) angles[i] = 0.0f;

    // --- Demo: send oscillating joint angles for 20 seconds ---
    for (int step = 0; step < 1000; step++) {
        float t = step * 0.02f;

        // Leg: sinusoidal motion (small amplitude)
        for (int i = 0; i < 6; i++) {
            angles[i] = 0.3f * sinf(t * 2.0f + i * 0.5f);
        }

        // Right arm: wave motion
        for (int i = 0; i < 7; i++) {
            angles[6 + i] = 0.5f * sinf(t * 1.5f + i * 0.7f);
        }

        // Left arm: wave motion (phase offset from right)
        for (int i = 0; i < 7; i++) {
            angles[13 + i] = 0.5f * sinf(t * 1.5f + i * 0.7f + 1.0f);
        }

        client.sendsvr(angles, (int)sizeof(angles));
        Sleep(20);
    }

    // Reset all joints to zero
    for (int i = 0; i < NUM_JOINTS; i++) angles[i] = 0.0f;
    client.sendsvr(angles, (int)sizeof(angles));

    return 0;
}