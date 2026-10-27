"""Consume each camera frame once, pairing chronological arrivals within a hard gate."""
from collections import deque


class StereoPairer:
    def __init__(self, max_skew_ms=10, capacity=6):
        self.queues=[deque(),deque()]
        self.capacity=capacity
        self.limit_ns=int(max_skew_ms*1e6)
        self.latest=None
        self.stats=dict(received=[0,0],matched_pairs=0,unmatched_drops=[0,0],overflow_drops=[0,0],arrival_gate_drops=[0,0], unconsumed_pair_drops=0)

    def push(self, eye, frame):
        # Caller serializes access. Frame=(image, source PTS, arrival monotonic ns, source clock ns).
        self.stats['received'][eye]+=1
        queue=self.queues[eye]
        if queue and (frame[2]<=queue[-1][2] or frame[3]<=queue[-1][3]):
            raise ValueError('Frame arrival timestamps must increase within each camera')
        queue.append(frame)
        if len(queue)>self.capacity:
            queue.popleft();self.stats['overflow_drops'][eye]+=1
        while all(self.queues):
            left,right=self.queues[0][0],self.queues[1][0]
            difference=left[3]-right[3]
            if abs(difference)<=self.limit_ns:
                pair=(self.queues[0].popleft(),self.queues[1].popleft())
                if abs(left[2]-right[2])<=self.limit_ns:
                    if self.latest is not None:self.stats['unconsumed_pair_drops']+=1
                    self.latest=pair
                    self.stats['matched_pairs']+=1
                else:
                    for eye in (0,1):self.stats['arrival_gate_drops'][eye]+=1
            else:
                eye=0 if difference<0 else 1
                self.queues[eye].popleft();self.stats['unmatched_drops'][eye]+=1

    def take_latest(self):
        pair=self.latest
        self.latest=None
        return pair
