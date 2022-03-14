﻿namespace SaxoRealTimeWorker
{
    public class WebSocketMessage
    {
        /// <summary>
        /// Message id that is unique within a streaming session.
        /// </summary>
        public long MessageId { get; set; }

        /// <summary>
        /// A unique subscription identifier. This is the reference id provided when the subscription was created.
        /// </summary>
        public string ReferenceId { get; set; }

        /// <summary>
        /// The message payload format. This is always Json, which has the value 0.
        /// </summary>
        public int PayloadFormat { get; set; }

        /// <summary>
        /// The payload is the actual data message. For Json this is a byte array representation of a UTF-8 encoded string.
        /// </summary>
        public byte[] Payload { get; set; }
    }
}
