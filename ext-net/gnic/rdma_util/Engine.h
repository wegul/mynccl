#pragma once
/**
 * @brief Engine is a class for send/recv threading. It will use JRing to communicate with producer/consumer.
 * For send, it acquires send-req from ring and push to SQ. For recv, it acquires recv-req from ring and push to RQ.
 * Hence, both thread are multiple-producer-single-consumer.
 */
class Engine {
};