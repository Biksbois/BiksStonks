using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Serilog;

using ILogger = Serilog.ILogger;

namespace SaxoRealTimeWorker
{
    public class SaxoWebSocket : IDisposable
    {
        private readonly string _contextId;
        private readonly string _referenceId;
        private readonly string _webSocketConnectionUrl;
        private readonly string _webSocketAuthorizationUrl;
        private readonly string _priceSubscriptionUrl;

        private ClientWebSocket _clientWebSocket;
        private CancellationTokenSource _cts;
        private Task _receiveTask;
        private string _token;
        private bool _disposed;
        private long _lastSeenMessageId;
        private long _receivedMessagesCount;

        public SaxoWebSocket()
        {
            //A valid OAuth2 _token - get a 24-hour token here: https://www.developer.saxo/openapi/token/current
            _token = "######";

            //Url for streaming server.
            _webSocketConnectionUrl = "wss://streaming.saxobank.com/sim/openapi/streamingws/connect";

            //Url for streaming server.
            _webSocketAuthorizationUrl = "https://streaming.saxobank.com/sim/openapi/streamingws/authorize";

            //Url for creating price subscription.
            _priceSubscriptionUrl = "https://gateway.saxobank.com/sim/openapi/trade/v1/prices/subscriptions";

            //A string provided by the client to correlate the stream and the subscription. Multiple subscriptions can use the same contextId.
            _contextId = "ctx_123_1";

            //A unique string provided by the client to identify a certain subscription in the stream.
            _referenceId = "rf_abc_1";
        }


        public async Task RunSample(CancellationTokenSource cts)
        {
            ThrowIfDisposed();

            _cts = cts;

            //First start the web socket connection.
            Task taskStartWebSocket = new Task(async () => { await StartWebSocket(); }, cts.Token);
            taskStartWebSocket.Start();

            //Then start the subscription.
            Task taskCreateSubscription = new Task(async () => { await CreateSubscription(); }, cts.Token);
            taskCreateSubscription.Start();

            //Start a task to renew the token when needed. If we don't do this the connection will be terminated once the token expires.
            DateTime tokenDummyExpiryTime = DateTime.Now.AddHours(2); //Here you need to provide the correct expiry time for the token. This is just a dummy value.
            //When the code breaks here, you probably need to add a valid _token in the code above.
            Task taskReauthorization = new Task(async () => { await ReauthorizeWhenNeeded(tokenDummyExpiryTime, cts.Token); }, cts.Token);
            taskReauthorization.Start();

            //Wait for both tasks to finish.
            Task[] tasks = { taskStartWebSocket, taskCreateSubscription, taskReauthorization };
            try
            {
                Task.WaitAll(tasks, cts.Token);
            }
            catch (OperationCanceledException)
            {
                return;
            }

            if (!cts.IsCancellationRequested) Console.WriteLine("Listening on web socket.");

            //Let's wait until someone stops the sample.
            while (!cts.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
            }
        }


        private async Task ReauthorizeWhenNeeded(DateTime tokenExpiryTime, CancellationToken cts)
        {
            //Renew the token a minute before it expires, to give us ample time to renew.
            TimeSpan tokenRenewalDelay = tokenExpiryTime.AddSeconds(-60).Subtract(DateTime.Now);

            while (!cts.IsCancellationRequested)
            {
                await Task.Delay(tokenRenewalDelay, cts);

                //This is where you should renew the token and get a new expiry time.
                //Here we have just created dummy values.
                tokenRenewalDelay = tokenRenewalDelay.Add(TimeSpan.FromHours(2));
                string refreshedToken = "<refreshedToken>";
                _token = refreshedToken;
                await Reauthorize(refreshedToken);
            }
        }


        private HttpClient CreateHttpClient()
        {
            var handler = new HttpClientHandler { AllowAutoRedirect = false, AutomaticDecompression = System.Net.DecompressionMethods.GZip };
            var httpClient = new HttpClient(handler);

            // Disable Expect: 100 Continue according to https://www.developer.saxo/openapi/learn/openapi-request-response
            // In our experience the same two-step process has been difficult to get to work reliable, especially as we support clients world wide, 
            // who connect to us through a multitude of network gateways and proxies.We also find that the actual bandwidth savings for the majority of API requests are limited, 
            // since most requests are quite small.
            // We therefore strongly recommend against using the Expect:100 - Continue header, and expect you to make sure your client library does not rely on this mechanism.
            // See: http://chrisoldwood.blogspot.com/2016/12/surprising-defaults-httpclient.html as an detailed example
            httpClient.DefaultRequestHeaders.ExpectContinue = false;

            return httpClient;
        }



        private async Task Reauthorize(string token)
        {
            using (HttpClient httpClient = CreateHttpClient())
            {
                Uri reauthorizationUrl = new Uri($"{_webSocketAuthorizationUrl}?contextid={_contextId}");
                using (HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Put, reauthorizationUrl))
                {
                    request.Headers.Authorization = new AuthenticationHeaderValue("BEARER", token);
                    HttpResponseMessage response = await httpClient.SendAsync(request, _cts.Token);
                    response.EnsureSuccessStatusCode();
                    Console.WriteLine("Refreshed token successfully and reauthorized.");
                }
            }
        }


        private async Task DeleteSubscription(string[] referenceIds)
        {
            ThrowIfDisposed();
            using (HttpClient httpClient = CreateHttpClient())
            {
                //In a real implementation we would look at the reference ids passed in and 
                //delete all the subscriptions listed. But in this implementation only one exists.
                string deleteSubscriptionUrl = $"{_priceSubscriptionUrl}/{_contextId}/{_referenceId}";
                using (HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Delete, deleteSubscriptionUrl))
                {
                    await httpClient.SendAsync(request, _cts.Token);
                }
            }
        }


        private async Task CreateSubscription()
        {
            ThrowIfDisposed();

            var subscriptionRequest = new
            {
                ContextId = _contextId,
                ReferenceId = _referenceId,
                Arguments = new
                {
                    AssetType = "FxSpot",
                    Uic = 21
                }
            };

            string json = JsonConvert.SerializeObject(subscriptionRequest);
            using (HttpClient httpClient = CreateHttpClient())
            {
                using (HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, _priceSubscriptionUrl))
                {
                    //Make sure you prepend the _token with the BEARER scheme
                    request.Headers.Authorization = new AuthenticationHeaderValue("BEARER", _token);
                    request.Content = new StringContent(json, Encoding.UTF8, "application/json");

                    try
                    {
                        HttpResponseMessage response = await httpClient.SendAsync(request, _cts.Token);
                        response.EnsureSuccessStatusCode();
                        // Saxobank is moving to HTTP/2, but here only HTTP/1.0 and HTTP/1.1 version requests are supported.
                        Console.WriteLine(request.RequestUri + " is using HTTP/" + response.Version);
                        // Read Response body
                        string responseBody = await response.Content.ReadAsStringAsync();
                        Console.WriteLine("Received snapshot:");
                        Console.WriteLine(JToken.Parse(responseBody).ToString(Formatting.Indented));
                        Console.WriteLine();
                    }
                    catch (TaskCanceledException)
                    {
                        return;
                    }
                    catch (HttpRequestException e)
                    {
                        Console.WriteLine("Subscription creation error.");
                        Console.WriteLine(e.Message);
                        _cts.Cancel(false);
                    }
                }
            }
        }


        private async Task StartWebSocket()
        {
            ThrowIfDisposed();

            //Make sure you append the contextId to the websocket connection url, and in the case of reconnection also the last seen message id.
            Uri url;
            if (_receivedMessagesCount > 0)
            {
                url = new Uri($"{_webSocketConnectionUrl}?contextid={_contextId}&messageid={_lastSeenMessageId}");
            }
            else
            {
                url = new Uri($"{_webSocketConnectionUrl}?contextid={_contextId}");
            }

            //Make sure you prepend the _token with the BEARER scheme
            string authorizationHeader = $"BEARER {_token}";

            //Connect to the web socket
            _clientWebSocket = new ClientWebSocket();
            _clientWebSocket.Options.SetRequestHeader("Authorization", authorizationHeader);

            try
            {
                await _clientWebSocket.ConnectAsync(url, _cts.Token);
            }
            catch (TaskCanceledException)
            {
                return;
            }
            catch (Exception e)
            {
                string flattenedExceptionMessages = FlattenExceptionMessages(e);
                Console.WriteLine("WebSocket connection error.");
                Console.WriteLine(flattenedExceptionMessages);
                _cts.Cancel(false);
                return;
            }

            //start listening for messages
            _receiveTask = ReceiveMessages(ErrorCallBack, SuccessCallBack, ControlMessageCallBack);
        }


        private async Task ErrorCallBack(Exception exception)
        {
            Console.WriteLine($"Error callback: {exception.Message}");
            if (exception is WebSocketException)
            {
                Console.WriteLine($"Reconnection with last seen message id {_lastSeenMessageId}.");
                _clientWebSocket?.Dispose();
                await StartWebSocket();
            }
            else
            {
                await StopWebSocket();
            }
        }


        private void SuccessCallBack(WebSocketMessage webSocketMessage)
        {
            PrintMessage(webSocketMessage);
        }


        private async Task ControlMessageCallBack(WebSocketMessage webSocketMessage)
        {
            //All control message reference ids start with an underscore
            if (!webSocketMessage.ReferenceId.StartsWith("_")) throw new ArgumentException($"Message {webSocketMessage.MessageId} with reference id {webSocketMessage.ReferenceId} is not a control message.");

            switch (webSocketMessage.ReferenceId)
            {
                case "_heartbeat":
                    // HeartBeat messages indicate that no new data is available. You do not need to do anything.
                    HeartbeatControlMessage[] heartBeatMessage = DecodeWebSocketMessagePayload<HeartbeatControlMessage[]>(webSocketMessage);
                    string referenceIdList = string.Join(",", heartBeatMessage.First().Heartbeats.Select(h => h.OriginatingReferenceId));
                    Console.WriteLine($"{webSocketMessage.MessageId}\tHeartBeat control message received for reference ids {referenceIdList}.");
                    break;
                case "_resetsubscriptions":
                    //For some reason the server is not able to send out messages,
                    //and needs the client to reset subscriptions by recreating them.
                    Console.WriteLine($"{webSocketMessage.MessageId}\tReset Subscription control message received.");
                    await ResetSubscriptions(webSocketMessage);
                    break;
                case "_disconnect":
                    //The server has disconnected the client. This messages requires you to re-authenticate
                    //if you wish to continue receiving messages. In this example we will just stop the WebSocket.
                    Console.WriteLine($"{webSocketMessage.MessageId}\tDisconnect control message received.");
                    await StopWebSocket();
                    break;
                default:
                    throw new ArgumentException($"Unknown control message reference id: {webSocketMessage.ReferenceId}");
            }
        }


        private async Task ResetSubscriptions(WebSocketMessage message)
        {
            ResetSubscriptionsControlMessage resetSubscriptionMessage = DecodeWebSocketMessagePayload<ResetSubscriptionsControlMessage>(message);

            //First delete the subscriptions the server tells us need to be reconnected.
            await DeleteSubscription(resetSubscriptionMessage.TargetReferenceIds);

            //Next create the subscriptions again.
            //You should keep track of a list of your subscriptions so you know which ones you have to recreate. 
            //Here we only have one subscription to illustrate the point.
            await CreateSubscription();
        }


        private async Task ReceiveMessages(Func<Exception, Task> errorCallback, Action<WebSocketMessage> successCallback, Func<WebSocketMessage, Task> controlMessageCallback)
        {
            try
            {
                //Create a buffer to hold the received messages in.
                byte[] buffer = new byte[16 * 1024];
                int offset = 0;
                Console.WriteLine("Start receiving messages.");

                //Listen while the socket is open.
                while (_clientWebSocket.State == WebSocketState.Open && !_disposed)
                {
                    ArraySegment<byte> receiveBuffer = new ArraySegment<byte>(buffer, offset, buffer.Length - offset);
                    WebSocketReceiveResult result = await _clientWebSocket.ReceiveAsync(receiveBuffer, _cts.Token);

                    if (_cts.IsCancellationRequested)
                        break;

                    switch (result.MessageType)
                    {
                        case WebSocketMessageType.Binary:
                            offset += result.Count;
                            if (result.EndOfMessage)
                            {
                                byte[] message = new byte[offset];
                                Array.Copy(buffer, message, offset);
                                offset = 0;
                                WebSocketMessage[] parsedMessages = ParseMessages(message);

                                foreach (WebSocketMessage parsedMessage in parsedMessages)
                                {
                                    //Be sure to cache the last seen message id
                                    _lastSeenMessageId = parsedMessage.MessageId;
                                    _receivedMessagesCount++;

                                    if (IsControlMessage(parsedMessage))
                                    {
                                        await controlMessageCallback(parsedMessage);
                                    }
                                    else
                                    {
                                        successCallback(parsedMessage);
                                    }
                                }
                            }

                            break;
                        case WebSocketMessageType.Close:
                            if (_clientWebSocket.State == WebSocketState.Open)
                                await _clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Goodbye",
                                    _cts.Token);
                            Console.WriteLine("Received a close frame.");
                            break;
                        case WebSocketMessageType.Text:
                            offset += result.Count;
                            await _clientWebSocket.CloseAsync(WebSocketCloseStatus.PolicyViolation, "Goodbye",
                                _cts.Token);
                            Console.WriteLine("Closing connection - Reason: received a text frame.");
                            break;
                    }
                }
            }
            catch (Exception e)
            {
                await errorCallback(e);
            }
        }


        public async Task StopWebSocket()
        {
            if (_disposed) return;

            try
            {
                //Send a close frame to the server.
                if (_clientWebSocket?.State == WebSocketState.Open)
                {
                    await _clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Goodbye", _cts.Token).ConfigureAwait(false);
                }

                //The server will respond with a close frame.
                //The close frame from the server might come after you have closed down your connection.

                if (null != _receiveTask) await _receiveTask;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }

            Console.WriteLine("Stopped receiving messages.");
        }


        private void PrintMessage(WebSocketMessage message)
        {
            //Extract the UTF8 encoded message payload.
            string messagePayload = Encoding.UTF8.GetString(message.Payload);
            Console.WriteLine($"{message.MessageId}\tPayload: {messagePayload}");
        }


        private WebSocketMessage[] ParseMessages(byte[] message)
        {
            List<WebSocketMessage> parsedMessages = new List<WebSocketMessage>();
            int index = 0;
            do
            {
                //First 8 bytes make up the message id. A 64 bit integer.
                long messageId = BitConverter.ToInt64(message, index);
                index += 8;

                //Skip the next two bytes that contain a reserved field.
                index += 2;

                //1 byte makes up the reference id length as an 8 bit integer. The reference id has a max length og 50 chars.
                byte referenceIdSize = message[index];
                index += 1;

                //n bytes make up the reference id. The reference id is an ASCII string.
                string referenceId = Encoding.ASCII.GetString(message, index, referenceIdSize);
                index += referenceIdSize;

                //1 byte makes up the payload format. The value 0 indicates that the payload format is Json.
                byte payloadFormat = message[index];
                index++;

                //4 bytes make up the payload length as a 32 bit integer. 
                int payloadSize = BitConverter.ToInt32(message, index);
                index += 4;

                //n bytes make up the actual payload. In the case of the payload format being Json, this is a UTF8 encoded string.
                byte[] payload = new byte[payloadSize];
                Array.Copy(message, index, payload, 0, payloadSize);
                index += payloadSize;

                WebSocketMessage parsedMessage = new WebSocketMessage
                {
                    MessageId = messageId,
                    ReferenceId = referenceId,
                    PayloadFormat = payloadFormat,
                    Payload = payload
                };

                parsedMessages.Add(parsedMessage);

            } while (index < message.Length);

            return parsedMessages.ToArray();
        }


        private bool IsControlMessage(WebSocketMessage webSocketMessage)
        {
            return webSocketMessage.ReferenceId.StartsWith("_");
        }


        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(SaxoWebSocket));
        }


        public void Dispose()
        {
            _clientWebSocket?.Dispose();
            _clientWebSocket = null;
            _disposed = true;
        }


        private string FlattenExceptionMessages(Exception exp)
        {
            string message = string.Empty;
            Exception innerException = exp;

            do
            {
                message = message + Environment.NewLine + (string.IsNullOrEmpty(innerException.Message) ? string.Empty : innerException.Message);
                innerException = innerException.InnerException;
            }
            while (innerException != null);

            if (message.Contains("409"))
                message += Environment.NewLine + "ContextId cannot be reused. Please create a new one and try again.";

            if (message.Contains("429"))
                message += Environment.NewLine + "You have made too many request. Please wait and try again.";

            return message;
        }

        private T DecodeWebSocketMessagePayload<T>(WebSocketMessage webSocketMessage)
        {
            string messagePayload = Encoding.UTF8.GetString(webSocketMessage.Payload);
            return JsonConvert.DeserializeObject<T>(messagePayload);
        }
    }
}
