using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace EurekaAddin
{
    public class ProcessInfo
    {
        public string id { get; set; }
        public string name { get; set; }
        public string description { get; set; }
        public int rule_count { get; set; }
    }

    public class RestClient
    {
        readonly HttpClient _client;
        readonly string _baseUrl;

        // Retry delays (ms) for 429 / 503 responses
        static readonly int[] _retryDelaysMs = { 1000, 2000, 4000 };

        public RestClient(string baseUrl = null)
        {
            // Allow override via environment variable; fall back to default port 8001
            _baseUrl = baseUrl
                ?? Environment.GetEnvironmentVariable("EUREKA_BACKEND_URL")
                ?? "http://localhost:8001";

            _client = new HttpClient { Timeout = TimeSpan.FromSeconds(90) };
            string apiKey = Environment.GetEnvironmentVariable("EUREKA_API_KEY") ?? "eureka-dev-key-change-me";
            _client.DefaultRequestHeaders.Add("X-API-Key", apiKey);
        }

        /// <summary>
        /// POST with automatic retry on HTTP 429 (rate-limited) and 503 (unavailable).
        /// Implements 3-attempt exponential backoff: 1s → 2s → 4s.
        /// </summary>
        async Task<HttpResponseMessage> PostWithRetry(string url, StringContent content)
        {
            HttpResponseMessage response = null;
            for (int attempt = 0; attempt <= _retryDelaysMs.Length; attempt++)
            {
                // Clone content for each attempt (HttpContent can only be sent once)
                var body = new StringContent(await content.ReadAsStringAsync(), Encoding.UTF8, "application/json");
                response = await _client.PostAsync(url, body);

                // 429 (TooManyRequests) not defined in .NET Framework 4.8 HttpStatusCode enum
                if ((int)response.StatusCode != 429 &&
                    response.StatusCode != HttpStatusCode.ServiceUnavailable)
                    break;

                if (attempt < _retryDelaysMs.Length)
                {
                    await Task.Delay(_retryDelaysMs[attempt]);
                }
            }
            return response;
        }

        private async Task<string> ReadStringUtf8(HttpContent content)
        {
            byte[] bytes = await content.ReadAsByteArrayAsync();
            return Encoding.UTF8.GetString(bytes);
        }

        public async Task<ValidationResult> ValidatePart(PartMetadata part)
        {
            string json = JsonConvert.SerializeObject(part);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await PostWithRetry($"{_baseUrl}/validate", content);
            response.EnsureSuccessStatusCode();
            string body = await ReadStringUtf8(response.Content);
            return JsonConvert.DeserializeObject<ValidationResult>(body);
        }

        public async Task<string> GetHealth()
        {
            // /health is exempt from API key — use a fresh client without the header
            using var plain = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var response = await plain.GetAsync($"{_baseUrl}/health");
            response.EnsureSuccessStatusCode();
            return await ReadStringUtf8(response.Content);
        }

        public async Task<List<ProcessInfo>> GetProcesses()
        {
            // /processes is also exempt from API key
            using var plain = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            var response = await plain.GetAsync($"{_baseUrl}/processes");
            response.EnsureSuccessStatusCode();
            string body = await ReadStringUtf8(response.Content);
            return JsonConvert.DeserializeObject<List<ProcessInfo>>(body);
        }

        public async Task<string> GenerateReport(ValidationResult result)
        {
            string json = JsonConvert.SerializeObject(result);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await PostWithRetry($"{_baseUrl}/report", content);
            response.EnsureSuccessStatusCode();
            string body = await ReadStringUtf8(response.Content);
            var obj = JsonConvert.DeserializeObject<Dictionary<string, string>>(body);
            return obj.ContainsKey("report") ? obj["report"] : "";
        }

        public async Task<byte[]> GeneratePdfReport(object payload)
        {
            string json = JsonConvert.SerializeObject(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await PostWithRetry($"{_baseUrl}/report/pdf", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }

        public async Task<string> SubmitFeedback(object feedbackPayload)
        {
            string json = JsonConvert.SerializeObject(feedbackPayload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await PostWithRetry($"{_baseUrl}/feedback", content);
            response.EnsureSuccessStatusCode();
            return await ReadStringUtf8(response.Content);
        }
    }
}
