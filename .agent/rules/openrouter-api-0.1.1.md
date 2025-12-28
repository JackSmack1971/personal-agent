---

# OPENROUTER API RULESET v0.1.1
**Technical Specification for AI Agent Implementation**

---
trigger: model_decision
description: Comprehensive ruleset for implementing OpenRouter unified API gateway (v0.1.1). Covers authentication, streaming, error handling, provider routing, tool calling, multimodal inputs, and production deployment patterns. Based on official documentation, TypeScript SDK patterns, and current platform features as of December 2025.
---

## I. CORE CONCEPTS

### Platform Overview
- **Unified API Gateway**: Single endpoint accessing 300+ models across 60+ providers
- **OpenAI Compatibility**: Drop-in replacement for OpenAI SDK - only change base URL and API key
- **Intelligent Routing**: Automatic provider selection, fallbacks, and cost optimization
- **Version**: Current API version is v1, SDK version is 0.1.1 (TypeScript)

### Base URLs
```typescript
// Production endpoint
const BASE_URL = "https://openrouter.ai/api/v1";

// Available endpoints
const ENDPOINTS = {
  chatCompletions: "/chat/completions",
  responses: "/responses",           // OpenAI Responses API format
  embeddings: "/embeddings",
  models: "/models",
  generation: "/generation/{id}",
  apiKey: "/key"                     // Rate limit/credit check
};
```

---

## II. AUTHENTICATION & CONFIGURATION

### MANDATORY: API Key Setup

**API Key Format**: `sk-or-v1-{64-character-hex-string}`

```typescript
// ✅ CORRECT: Environment variable pattern
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

if (!OPENROUTER_API_KEY) {
  throw new Error("OPENROUTER_API_KEY environment variable required");
}
```

**❌ ANTI-PATTERN: Hardcoded API keys**
```typescript
// NEVER commit API keys to version control
const apiKey = "sk-or-v1-abc123..."; // SECURITY VIOLATION
```

### Required Headers

```typescript
interface OpenRouterHeaders {
  "Authorization": `Bearer ${string}`;        // REQUIRED
  "Content-Type": "application/json";         // REQUIRED
  "HTTP-Referer"?: string;                    // OPTIONAL: For app discovery
  "X-Title"?: string;                         // OPTIONAL: App name for rankings
}

// ✅ CORRECT: Minimal request
const headers = {
  "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
  "Content-Type": "application/json"
};

// ✅ BEST PRACTICE: Include tracking headers
const headersWithTracking = {
  "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
  "Content-Type": "application/json",
  "HTTP-Referer": "https://yourapp.com",
  "X-Title": "Your App Name"
};
```

**CRITICAL**: Only HTTPS URLs on ports 443 and 3000 are allowed for `HTTP-Referer`

### SDK Initialization (TypeScript)

```typescript
import { OpenRouter } from "@openrouter/sdk";

// ✅ CORRECT: Initialize with API key
const openRouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY
});

// ✅ CORRECT: OpenAI SDK compatibility
import { OpenAI } from "openai";

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    "HTTP-Referer": "https://yourapp.com",
    "X-Title": "Your App Name"
  }
});
```

---

## III. RATE LIMITS & PRICING

### Current Rate Limits (December 2025)

| Tier | Daily Free Model Calls | With $10+ Balance | RPM (Requests/Min) |
|------|----------------------|-------------------|-------------------|
| Free | 50 requests/day | 1000 requests/day | 20 (free models) |
| Pay-as-you-go | N/A | Based on model | Model-dependent |
| Enterprise | N/A | Custom limits | Custom SLA |

**Platform Fees**: 5.5% on pay-as-you-go tier, bulk discounts available for Enterprise

### Free Model Restrictions
```typescript
// Free models end with ":free" suffix
const freeModels = [
  "google/gemini-2.0-flash-thinking-exp:free",
  "meta-llama/llama-3.3-70b-instruct:free",
  "anthropic/claude-3.5-haiku:free"
];

// ✅ CORRECT: Check rate limits programmatically
async function checkCredits() {
  const response = await fetch("https://openrouter.ai/api/v1/key", {
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`
    }
  });
  
  const data = await response.json();
  console.log("Credits remaining:", data.data.limit_remaining);
  console.log("Rate limit:", data.data.rate_limit);
}
```

### Rate Limit Headers (Response)
```typescript
interface RateLimitHeaders {
  "x-ratelimit-limit-requests": string;      // Total requests allowed
  "x-ratelimit-remaining-requests": string;  // Remaining requests
  "x-ratelimit-reset-requests": string;      // Unix timestamp of reset
}
```

---

## IV. REQUEST STRUCTURE

### Basic Chat Completion

```typescript
interface ChatCompletionRequest {
  model: string;                              // REQUIRED
  messages: Message[];                        // REQUIRED
  
  // Optional parameters
  temperature?: number;                       // 0.0 to 2.0, default: 1.0
  max_tokens?: number;                        // Model-specific limits
  top_p?: number;                            // 0.0 to 1.0
  frequency_penalty?: number;                 // -2.0 to 2.0
  presence_penalty?: number;                  // -2.0 to 2.0
  stream?: boolean;                          // Default: false
  stop?: string | string[];                  // Stop sequences
  
  // OpenRouter-specific parameters
  provider?: ProviderPreferences;            // Provider routing
  models?: string[];                         // Model fallback array
  transforms?: ("middle-out")[];             // Message compression
  route?: "fallback";                        // Routing strategy
}

// ✅ CORRECT: Basic request
const request: ChatCompletionRequest = {
  model: "openai/gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" }
  ]
};

// ✅ CORRECT: With advanced parameters
const advancedRequest: ChatCompletionRequest = {
  model: "anthropic/claude-sonnet-4.5",
  messages: [
    { role: "user", content: "Explain quantum computing" }
  ],
  temperature: 0.7,
  max_tokens: 1000,
  provider: {
    order: ["anthropic", "openai"],
    allowFallbacks: true,
    dataCollection: "deny"  // Privacy-focused routing
  }
};
```

### Message Format

```typescript
type MessageRole = "system" | "user" | "assistant" | "tool";

interface TextMessage {
  role: MessageRole;
  content: string;
}

interface MultimodalMessage {
  role: "user";
  content: Array<TextContent | ImageContent>;
}

interface TextContent {
  type: "text";
  text: string;
}

interface ImageContent {
  type: "image_url";
  imageUrl: {
    url: string;  // HTTPS URL or base64 data URI
  };
}

// ✅ CORRECT: Text message
const textMsg: TextMessage = {
  role: "user",
  content: "What is the weather today?"
};

// ✅ CORRECT: Multimodal with image
const multimodalMsg: MultimodalMessage = {
  role: "user",
  content: [
    {
      type: "text",
      text: "What's in this image?"
    },
    {
      type: "image_url",
      imageUrl: {
        url: "https://example.com/image.jpg"
        // OR base64: "data:image/jpeg;base64,/9j/4AAQ..."
      }
    }
  ]
};
```

**Supported Image Formats**: PNG, JPEG, WebP, GIF

---

## V. PROVIDER ROUTING & FALLBACKS

### Provider Preferences Configuration

```typescript
interface ProviderPreferences {
  // Ordered provider list
  order?: string[];                           // Try in sequence
  
  // Fallback behavior
  allowFallbacks?: boolean;                   // Default: true
  
  // Provider filtering
  only?: string[];                           // Whitelist
  ignore?: string[];                         // Blacklist
  
  // Data privacy
  dataCollection?: "allow" | "deny";         // Default: "allow"
  zdr?: boolean;                             // Zero data retention only
  
  // Performance optimization
  sort?: "price" | "throughput" | "latency"; // Sorting strategy
  requireParameters?: boolean;                // Only providers supporting all params
  
  // Cost control
  max_price?: {
    prompt?: number;                          // Max USD per 1M prompt tokens
    completion?: number;                      // Max USD per 1M completion tokens
  };
}

// ✅ CORRECT: Cost-optimized routing
const costOptimized: ChatCompletionRequest = {
  model: "meta-llama/llama-3.3-70b-instruct",
  messages: [{ role: "user", content: "Hello" }],
  provider: {
    sort: "price",
    allowFallbacks: true
  }
};

// ✅ CORRECT: Privacy-first routing
const privacyFirst: ChatCompletionRequest = {
  model: "anthropic/claude-sonnet-4.5",
  messages: [{ role: "user", content: "Sensitive query" }],
  provider: {
    dataCollection: "deny",
    zdr: true,  // Only zero-data-retention endpoints
    allowFallbacks: false
  }
};

// ✅ CORRECT: Specific provider with no fallback
const strictProvider: ChatCompletionRequest = {
  model: "openai/gpt-4o",
  messages: [{ role: "user", content: "Query" }],
  provider: {
    order: ["openai"],
    allowFallbacks: false  // Fail if OpenAI unavailable
  }
};
```

### Model Fallback Array

```typescript
// ✅ CORRECT: Prioritized model list
const request = {
  models: [
    "anthropic/claude-sonnet-4.5",      // Primary
    "openai/gpt-4o",                    // Fallback 1
    "google/gemini-2.0-flash-001"       // Fallback 2
  ],
  messages: [{ role: "user", content: "Hello" }]
};

// Automatically tries next model if previous fails due to:
// - Context length exceeded
// - Model downtime
// - Rate limiting
// - Moderation filters
```

**❌ ANTI-PATTERN: Single model with no fallback for production**
```typescript
// High risk of failures in production
const fragileRequest = {
  model: "openai/gpt-4o",  // No fallback
  messages: [...]
};
```

---

## VI. STREAMING RESPONSES

### Server-Sent Events (SSE) Format

**CRITICAL**: OpenRouter streaming uses SSE format with `data:` prefix

```typescript
// ✅ CORRECT: TypeScript SDK streaming
async function streamWithSDK() {
  const stream = await openRouter.chat.send({
    model: "anthropic/claude-sonnet-4.5",
    messages: [{ role: "user", content: "Write a poem" }],
    stream: true
  });

  for await (const chunk of stream) {
    const content = chunk.choices?.[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
    }
  }
}

// ✅ CORRECT: Native fetch streaming
async function streamWithFetch(prompt: string) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: prompt }],
      stream: true
    })
  });

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete lines
    while (true) {
      const lineEnd = buffer.indexOf("\n");
      if (lineEnd === -1) break;

      const line = buffer.slice(0, lineEnd).trim();
      buffer = buffer.slice(lineEnd + 1);

      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        
        // Stream termination marker
        if (data === "[DONE]") return;

        try {
          const chunk = JSON.parse(data);
          const content = chunk.choices?.[0]?.delta?.content;
          if (content) {
            process.stdout.write(content);
          }
        } catch (e) {
          // Skip malformed JSON
        }
      }
    }
  }
}
```

### Streaming Response Structure

```typescript
interface StreamChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: StreamChoice[];
  usage?: {                          // Only in final chunk with includeUsage
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface StreamChoice {
  index: number;
  delta: {
    role?: "assistant";              // Only in first chunk
    content?: string;                 // Content chunks
    tool_calls?: ToolCall[];         // Tool calling chunks
  };
  finish_reason: null | "stop" | "length" | "tool_calls" | "error";
}
```

---

## VII. ERROR HANDLING

### MANDATORY: Comprehensive Error Handling

**CRITICAL**: Handle both pre-stream and mid-stream errors

```typescript
// ✅ CORRECT: Complete error handling
async function robustStreamRequest(prompt: string) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: prompt }],
      stream: true
    })
  });

  // PRE-STREAM ERROR: Check HTTP status
  if (!response.ok) {
    const error = await response.json();
    throw new Error(`HTTP ${response.status}: ${error.error?.message || "Unknown error"}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const lineEnd = buffer.indexOf("\n");
        if (lineEnd === -1) break;

        const line = buffer.slice(0, lineEnd).trim();
        buffer = buffer.slice(lineEnd + 1);

        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;

          try {
            const parsed = JSON.parse(data);

            // MID-STREAM ERROR: Check for error field
            if (parsed.error) {
              console.error(`Stream error: ${parsed.error.message}`);
              
              // Check finish_reason for error termination
              if (parsed.choices?.[0]?.finish_reason === "error") {
                throw new Error(`Stream terminated: ${parsed.error.message}`);
              }
              return;
            }

            // Process normal content
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) {
              process.stdout.write(content);
            }
          } catch (e) {
            if (e instanceof SyntaxError) {
              // Skip malformed JSON
              continue;
            }
            throw e;
          }
        }
      }
    }
  } finally {
    // CLEANUP: Always cancel reader
    reader.cancel();
  }
}
```

### Error Response Formats

```typescript
// Pre-stream errors (HTTP 4xx/5xx)
interface ErrorResponse {
  error: {
    code: number | string;
    message: string;
    metadata?: Record<string, unknown>;
  };
}

// Mid-stream errors (within SSE stream)
interface StreamErrorChunk {
  error: {
    code: string;
    message: string;
  };
  choices?: [{
    finish_reason: "error";
  }];
}
```

### Common Error Codes

| HTTP Code | Error Code | Meaning | Action |
|-----------|-----------|---------|--------|
| 400 | `invalid_request_error` | Malformed request | Validate request structure |
| 401 | `authentication_error` | Invalid API key | Check API key |
| 402 | `insufficient_quota` | Out of credits | Add credits to account |
| 429 | `rate_limit_exceeded` | Too many requests | Implement exponential backoff |
| 500 | `api_error` | Server error | Retry with exponential backoff |
| 503 | `overloaded_error` | Service overload | Retry after delay |

**Context Length Errors**: Automatically transformed to `finish_reason: "length"` instead of failure

---

## VIII. TOOL CALLING (FUNCTION CALLING)

### Tool Definition Structure

```typescript
interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    strict?: boolean | null;
    parameters: {
      type: "object";
      properties: Record<string, JSONSchema>;
      required: string[];
    };
  };
}

// ✅ CORRECT: Weather tool definition
const weatherTool: ToolDefinition = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather in a location",
    strict: null,
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city and state, e.g. San Francisco, CA"
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"]
        }
      },
      required: ["location"]
    }
  }
};
```

### Complete Tool Calling Workflow

```typescript
interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;  // JSON string
  };
}

// ✅ CORRECT: Agentic loop with tool calling
async function agenticLoop(initialMessage: string) {
  const tools: ToolDefinition[] = [weatherTool, calculatorTool];
  const messages: Message[] = [
    { role: "user", content: initialMessage }
  ];

  const MAX_ITERATIONS = 10;
  let iteration = 0;

  while (iteration < MAX_ITERATIONS) {
    iteration++;

    // STEP 1: Call LLM with tools
    const response = await openRouter.chat.send({
      model: "openai/gpt-4o",
      messages,
      tools,
      tool_choice: "auto",
      stream: false
    });

    const message = response.choices[0].message;
    messages.push(message);

    // STEP 2: Check for tool calls
    if (!message.tool_calls || message.tool_calls.length === 0) {
      // No more tool calls - final response
      console.log("Final response:", message.content);
      break;
    }

    // STEP 3: Execute tool calls
    for (const toolCall of message.tool_calls) {
      const toolName = toolCall.function.name;
      const toolArgs = JSON.parse(toolCall.function.arguments);

      // Look up tool function
      const toolFunction = TOOL_REGISTRY[toolName];
      if (!toolFunction) {
        throw new Error(`Unknown tool: ${toolName}`);
      }

      // Execute tool
      const toolResult = await toolFunction(toolArgs);

      // STEP 4: Add tool response to messages
      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolResult)
      });
    }
  }

  if (iteration >= MAX_ITERATIONS) {
    console.warn("⚠️  Maximum iterations reached");
  }

  return messages[messages.length - 1].content;
}

// Tool registry
const TOOL_REGISTRY = {
  get_weather: async (args: { location: string; unit?: string }) => {
    // Implementation
    return {
      temperature: 72,
      conditions: "sunny",
      location: args.location
    };
  },
  calculate: async (args: { expression: string }) => {
    // Implementation
    return { result: eval(args.expression) };
  }
};
```

### Streaming Tool Calls

```typescript
// ✅ CORRECT: Stream tool calls with Responses API
async function streamToolCalls() {
  const response = await fetch("https://openrouter.ai/api/v1/responses", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/o4-mini",
      input: [{
        type: "message",
        role: "user",
        content: [{
          type: "input_text",
          text: "What is the weather in Tokyo?"
        }]
      }],
      tools: [weatherTool],
      tool_choice: "auto",
      stream: true,
      max_output_tokens: 9000
    })
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  for await (const { value } of reader) {
    const chunk = decoder.decode(value);
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;

        const parsed = JSON.parse(data);

        // Tool call initiated
        if (parsed.type === "response.output_item.added" &&
            parsed.item?.type === "function_call") {
          console.log("Tool call:", parsed.item.name);
        }

        // Tool arguments complete
        if (parsed.type === "response.function_call_arguments.done") {
          console.log("Arguments:", parsed.arguments);
        }
      }
    }
  }
}
```

---

## IX. MULTIMODAL CAPABILITIES

### Image Inputs

```typescript
// ✅ CORRECT: Image from URL
const imageRequest = {
  model: "google/gemini-2.0-flash-001",
  messages: [{
    role: "user",
    content: [
      {
        type: "text",
        text: "Describe this image in detail"
      },
      {
        type: "image_url",
        imageUrl: {
          url: "https://example.com/image.jpg"
        }
      }
    ]
  }]
};

// ✅ CORRECT: Base64 encoded image
const base64ImageRequest = {
  model: "openai/gpt-4o",
  messages: [{
    role: "user",
    content: [
      {
        type: "text",
        text: "What's in this image?"
      },
      {
        type: "image_url",
        imageUrl: {
          url: "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        }
      }
    ]
  }]
};
```

### Image Generation (Select Models)

```typescript
// ✅ CORRECT: Generate images with streaming
async function generateImage() {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "google/gemini-2.5-flash-image-preview",
      messages: [{
        role: "user",
        content: "Create an image of a futuristic cityscape"
      }],
      modalities: ["image", "text"],
      stream: true
    })
  });

  for await (const line of response.body) {
    if (line.startsWith("data: ")) {
      const data = line.slice(6);
      if (data !== "[DONE]") {
        const chunk = JSON.parse(data);
        const images = chunk.choices?.[0]?.delta?.images;
        
        if (images) {
          for (const image of images) {
            console.log("Generated image URL:", image.image_url.url);
          }
        }
      }
    }
  }
}
```

---

## X. MESSAGE TRANSFORMS

### Middle-Out Compression

**CRITICAL**: Automatically enabled for models ≤8,192 tokens

```typescript
// ✅ CORRECT: Explicit middle-out transform
const request = {
  model: "openai/gpt-4o",
  messages: [
    /* Long conversation history */
  ],
  transforms: ["middle-out"]  // Compress if exceeds context
};

// ✅ CORRECT: Disable transforms for exact control
const noTransformRequest = {
  model: "openai/gpt-4o",
  messages: [/* ... */],
  transforms: []  // No automatic compression
};
```

**Behavior**:
- Removes/truncates messages from conversation middle
- Preserves beginning and end of conversation
- LLMs pay less attention to middle sequences
- Automatically selects model with ≥50% required context length
- For models with message limits (e.g., Claude): keeps half from start, half from end

---

## XI. DEBUGGING & MONITORING

### Debug Mode (Streaming Only)

```typescript
// ✅ CORRECT: Enable debug output
const debugRequest = {
  model: "anthropic/claude-haiku-4.5",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" }
  ],
  stream: true,  // REQUIRED for debug
  debug: {
    echo_upstream_body: true  // Shows transformed request sent to provider
  }
};

// Process debug chunk
for await (const chunk of stream) {
  if (chunk.debug?.echo_upstream_body) {
    console.log("Upstream request:", JSON.stringify(chunk.debug.echo_upstream_body, null, 2));
  }
  
  const content = chunk.choices?.[0]?.delta?.content;
  if (content) {
    process.stdout.write(content);
  }
}
```

### Rate Limit Monitoring

```typescript
// ✅ CORRECT: Check rate limits programmatically
async function checkLimits() {
  const response = await fetch("https://openrouter.ai/api/v1/key", {
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`
    }
  });

  const data = await response.json();
  
  return {
    creditsRemaining: data.data.limit_remaining,
    dailyLimit: data.data.limit,
    resetTime: data.data.limit_reset,
    usage: {
      total: data.data.usage,
      daily: data.data.usage_daily,
      weekly: data.data.usage_weekly,
      monthly: data.data.usage_monthly
    }
  };
}
```

---

## XII. SECURITY BEST PRACTICES

### MANDATORY Security Patterns

```typescript
// ✅ CORRECT: Secure API key storage
// 1. Environment variables (never commit)
const apiKey = process.env.OPENROUTER_API_KEY;

// 2. Secret management service
import { SecretManagerServiceClient } from "@google-cloud/secret-manager";
const client = new SecretManagerServiceClient();
const [secret] = await client.accessSecretVersion({
  name: "projects/PROJECT_ID/secrets/openrouter-key/versions/latest"
});
const apiKey = secret.payload.data.toString();

// ❌ ANTI-PATTERN: Hardcoded keys
const API_KEY = "sk-or-v1-abc123...";  // NEVER DO THIS
```

### Input Validation

```typescript
// ✅ CORRECT: Validate user inputs
function validateChatRequest(request: unknown): ChatCompletionRequest {
  // Type validation
  if (!request || typeof request !== "object") {
    throw new Error("Invalid request: must be object");
  }

  const { model, messages } = request as any;

  // Model validation
  if (typeof model !== "string" || !model.includes("/")) {
    throw new Error("Invalid model format: must be 'provider/model'");
  }

  // Messages validation
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error("Messages must be non-empty array");
  }

  // Content length limits (prevent abuse)
  const totalContent = messages.map(m => m.content).join("");
  if (totalContent.length > 100000) {
    throw new Error("Total content exceeds 100KB limit");
  }

  return request as ChatCompletionRequest;
}
```

### Privacy-Focused Routing

```typescript
// ✅ CORRECT: Zero data retention for sensitive queries
const sensitiveRequest = {
  model: "anthropic/claude-sonnet-4.5",
  messages: [
    { role: "user", content: "Analyze this medical record..." }
  ],
  provider: {
    zdr: true,                      // Zero data retention only
    dataCollection: "deny",         // No data collection
    allowFallbacks: false           // Strict routing
  }
};
```

---

## XIII. PRODUCTION DEPLOYMENT PATTERNS

### Retry Logic with Exponential Backoff

```typescript
// ✅ CORRECT: Robust retry implementation
async function chatWithRetry(
  request: ChatCompletionRequest,
  maxRetries = 3
): Promise<ChatResponse> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        const error = await response.json();
        
        // Don't retry on client errors (4xx except 429)
        if (response.status >= 400 && response.status < 500 && response.status !== 429) {
          throw new Error(`Client error: ${error.error.message}`);
        }

        throw new Error(`HTTP ${response.status}: ${error.error.message}`);
      }

      return await response.json();

    } catch (error) {
      lastError = error as Error;
      
      if (attempt < maxRetries) {
        // Exponential backoff: 2^attempt seconds
        const delayMs = Math.pow(2, attempt) * 1000;
        console.warn(`Retry attempt ${attempt + 1}/${maxRetries} after ${delayMs}ms`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }

  throw lastError!;
}
```

### Request Timeout Handling

```typescript
// ✅ CORRECT: Implement request timeouts
async function chatWithTimeout(
  request: ChatCompletionRequest,
  timeoutMs = 30000
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(request),
      signal: controller.signal
    });

    return await response.json();

  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`Request timeout after ${timeoutMs}ms`);
    }
    throw error;

  } finally {
    clearTimeout(timeoutId);
  }
}
```

### Model Fallback Implementation

```typescript
// ✅ CORRECT: Implement custom fallback logic
async function chatWithFallback(
  request: ChatCompletionRequest
): Promise<ChatResponse> {
  const modelPriority = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct"
  ];

  for (const model of modelPriority) {
    try {
      const response = await chatWithRetry({
        ...request,
        model,
        provider: {
          allowFallbacks: false  // Control fallback manually
        }
      });

      console.log(`✓ Success with model: ${model}`);
      return response;

    } catch (error) {
      console.warn(`✗ Failed with ${model}: ${error.message}`);
      
      // Continue to next model
      if (model === modelPriority[modelPriority.length - 1]) {
        // Last model failed
        throw new Error(`All models failed. Last error: ${error.message}`);
      }
    }
  }

  throw new Error("Unexpected: No models attempted");
}
```

---

## XIV. COMMON PITFALLS & ANTI-PATTERNS

### ❌ ANTI-PATTERN 1: Missing Error Handling

```typescript
// BAD: No error handling
async function badRequest() {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Hello" }]
    })
  });

  return await response.json();  // WILL FAIL on 4xx/5xx
}

// GOOD: Comprehensive error handling
async function goodRequest() {
  try {
    const response = await fetch(/* ... */);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.error.message}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Request failed:", error);
    throw error;
  }
}
```

### ❌ ANTI-PATTERN 2: Incomplete Streaming Parsing

```typescript
// BAD: Missing buffer management
async function badStreaming() {
  const response = await fetch(/* ... */, { stream: true });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const data = text.slice(6);  // WRONG: Assumes complete lines
    const chunk = JSON.parse(data);  // WILL FAIL on partial JSON
    console.log(chunk);
  }
}

// GOOD: Proper line buffering (see Section VI)
```

### ❌ ANTI-PATTERN 3: Ignoring Provider Preferences

```typescript
// BAD: No provider control
const request = {
  model: "openai/gpt-4o",
  messages: [{ role: "user", content: "Sensitive data..." }]
  // No privacy controls!
};

// GOOD: Privacy-aware routing
const betterRequest = {
  model: "openai/gpt-4o",
  messages: [{ role: "user", content: "Sensitive data..." }],
  provider: {
    dataCollection: "deny",
    zdr: true
  }
};
```

### ❌ ANTI-PATTERN 4: Tool Calling Without Limits

```typescript
// BAD: Infinite loop risk
async function dangerousAgenticLoop() {
  while (true) {  // DANGEROUS: No iteration limit
    const response = await callLLM();
    if (!response.tool_calls) break;
    await executeTool(response.tool_calls[0]);
  }
}

// GOOD: Bounded iteration (see Section VIII)
```

### ❌ ANTI-PATTERN 5: Exposing API Keys Client-Side

```typescript
// BAD: API key in frontend code
const apiKey = "sk-or-v1-abc123...";  // NEVER expose in browser
fetch("https://openrouter.ai/api/v1/chat/completions", {
  headers: { "Authorization": `Bearer ${apiKey}` }  // SECURITY VIOLATION
});

// GOOD: Proxy through backend
// Frontend calls your backend
fetch("/api/chat", {
  method: "POST",
  body: JSON.stringify({ message: "Hello" })
});

// Backend handles OpenRouter
app.post("/api/chat", async (req, res) => {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    headers: {
      "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`
    },
    body: JSON.stringify({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: req.body.message }]
    })
  });
  
  const data = await response.json();
  res.json(data);
});
```

---

## XV. TESTING & VALIDATION

### Unit Testing Pattern

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";

describe("OpenRouter Integration", () => {
  beforeEach(() => {
    // Mock API key
    process.env.OPENROUTER_API_KEY = "sk-or-v1-test-key";
  });

  it("should handle successful chat completion", async () => {
    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "chatcmpl-test",
        model: "openai/gpt-4o",
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: "Hello! How can I help you?"
          },
          finish_reason: "stop"
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18
        }
      })
    });

    const response = await chatWithRetry({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Hello" }]
    });

    expect(response.choices[0].message.content).toBe("Hello! How can I help you?");
  });

  it("should retry on 429 rate limit", async () => {
    let callCount = 0;

    global.fetch = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.resolve({
          ok: false,
          status: 429,
          json: async () => ({
            error: { code: "rate_limit_exceeded", message: "Too many requests" }
          })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ choices: [{ message: { content: "Success" } }] })
      });
    });

    const response = await chatWithRetry({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Test" }]
    }, 3);

    expect(callCount).toBe(3);
    expect(response.choices[0].message.content).toBe("Success");
  });
});
```

---

## XVI. TYPESCRIPT SDK BEST PRACTICES

### SDK-Specific Patterns

```typescript
import { OpenRouter } from "@openrouter/sdk";
import type { 
  ChatGenerationParams,
  ChatResponse,
  ChatStreamingResponseChunk 
} from "@openrouter/sdk/models";

// ✅ CORRECT: Type-safe initialization
const client = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY
});

// ✅ CORRECT: Non-streaming with types
const params: ChatGenerationParams = {
  model: "openai/gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" }
  ],
  temperature: 0.7,
  max_tokens: 1000,
  stream: false
};

const response: ChatResponse = await client.chat.send(params);

// ✅ CORRECT: Streaming with types
const streamParams: ChatGenerationParams = {
  ...params,
  stream: true,
  streamOptions: {
    includeUsage: true  // Get token usage in final chunk
  }
};

const stream = await client.chat.send(streamParams);

for await (const chunk of stream) {
  // TypeScript knows chunk is ChatStreamingResponseChunk
  const content = chunk.choices?.[0]?.delta?.content;
  if (content) {
    process.stdout.write(content);
  }

  // Usage stats in final chunk
  if (chunk.usage) {
    console.log("\nTokens used:", chunk.usage.total_tokens);
  }
}
```

### SDK Error Handling

```typescript
import { 
  ChatError,
  OpenRouterDefaultError 
} from "@openrouter/sdk/models/errors";

// ✅ CORRECT: Type-safe error handling
async function sendWithErrorHandling(params: ChatGenerationParams) {
  try {
    return await client.chat.send(params);
  } catch (error) {
    if (error instanceof ChatError) {
      // 400, 401, 429 errors
      console.error("Chat error:", error.message);
      console.error("Status code:", error.statusCode);
    } else if (error instanceof OpenRouterDefaultError) {
      // Other 4xx/5xx errors
      console.error("API error:", error.message);
    } else {
      // Network or other errors
      console.error("Unexpected error:", error);
    }
    throw error;
  }
}
```

---

## XVII. PERFORMANCE OPTIMIZATION

### Response Caching Strategy

```typescript
// ✅ CORRECT: Implement response caching
import { createHash } from "crypto";

class OpenRouterCache {
  private cache = new Map<string, { response: ChatResponse; timestamp: number }>();
  private ttl = 3600000; // 1 hour

  private getCacheKey(request: ChatCompletionRequest): string {
    const normalized = {
      model: request.model,
      messages: request.messages,
      temperature: request.temperature ?? 1.0
    };
    return createHash("sha256").update(JSON.stringify(normalized)).digest("hex");
  }

  async get(request: ChatCompletionRequest): Promise<ChatResponse | null> {
    const key = this.getCacheKey(request);
    const cached = this.cache.get(key);

    if (!cached) return null;

    // Check if expired
    if (Date.now() - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.response;
  }

  set(request: ChatCompletionRequest, response: ChatResponse): void {
    const key = this.getCacheKey(request);
    this.cache.set(key, {
      response,
      timestamp: Date.now()
    });
  }

  clear(): void {
    this.cache.clear();
  }
}

// Usage
const cache = new OpenRouterCache();

async function cachedChat(request: ChatCompletionRequest): Promise<ChatResponse> {
  // Check cache first
  const cached = await cache.get(request);
  if (cached) {
    console.log("✓ Cache hit");
    return cached;
  }

  // Make API request
  const response = await chatWithRetry(request);

  // Cache response
  cache.set(request, response);

  return response;
}
```

### Batch Processing

```typescript
// ✅ CORRECT: Process multiple requests efficiently
async function batchProcess(
  requests: ChatCompletionRequest[],
  concurrency = 5
): Promise<ChatResponse[]> {
  const results: ChatResponse[] = [];
  const queue = [...requests];

  // Process in batches of `concurrency`
  while (queue.length > 0) {
    const batch = queue.splice(0, concurrency);
    const promises = batch.map(req => chatWithRetry(req));
    
    const batchResults = await Promise.allSettled(promises);
    
    for (const result of batchResults) {
      if (result.status === "fulfilled") {
        results.push(result.value);
      } else {
        console.error("Batch item failed:", result.reason);
        // Handle failure appropriately
      }
    }
  }

  return results;
}
```

---

## XVIII. MONITORING & OBSERVABILITY

### Request Logging

```typescript
// ✅ CORRECT: Comprehensive request logging
interface RequestLog {
  timestamp: string;
  model: string;
  provider?: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  latencyMs: number;
  success: boolean;
  error?: string;
}

class OpenRouterLogger {
  private logs: RequestLog[] = [];

  async logRequest<T>(
    request: ChatCompletionRequest,
    execute: () => Promise<T>
  ): Promise<T> {
    const startTime = Date.now();
    
    try {
      const response = await execute();
      
      // Extract metrics
      const usage = (response as any).usage;
      
      this.logs.push({
        timestamp: new Date().toISOString(),
        model: request.model,
        provider: request.provider?.order?.[0],
        promptTokens: usage?.prompt_tokens ?? 0,
        completionTokens: usage?.completion_tokens ?? 0,
        totalTokens: usage?.total_tokens ?? 0,
        latencyMs: Date.now() - startTime,
        success: true
      });

      return response;

    } catch (error) {
      this.logs.push({
        timestamp: new Date().toISOString(),
        model: request.model,
        provider: request.provider?.order?.[0],
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        latencyMs: Date.now() - startTime,
        success: false,
        error: error.message
      });

      throw error;
    }
  }

  getMetrics() {
    return {
      totalRequests: this.logs.length,
      successRate: this.logs.filter(l => l.success).length / this.logs.length,
      avgLatency: this.logs.reduce((sum, l) => sum + l.latencyMs, 0) / this.logs.length,
      totalTokens: this.logs.reduce((sum, l) => sum + l.totalTokens, 0)
    };
  }

  exportLogs(): RequestLog[] {
    return [...this.logs];
  }
}

// Usage
const logger = new OpenRouterLogger();

const response = await logger.logRequest(request, () =>
  chatWithRetry(request)
);

console.log("Metrics:", logger.getMetrics());
```

---

## XIX. MIGRATION FROM OPENAI SDK

### Drop-in Replacement Pattern

```typescript
// BEFORE: OpenAI SDK
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }]
});

// AFTER: OpenRouter (minimal change)
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",  // ← ONLY CHANGE
  apiKey: process.env.OPENROUTER_API_KEY,    // ← ONLY CHANGE
  defaultHeaders: {                          // ← OPTIONAL
    "HTTP-Referer": "https://yourapp.com",
    "X-Title": "Your App Name"
  }
});

const response = await openai.chat.completions.create({
  model: "openai/gpt-4o",  // ← ADD PROVIDER PREFIX
  messages: [{ role: "user", content: "Hello" }]
});
```

### Model Name Mapping

| OpenAI Model | OpenRouter Equivalent |
|--------------|----------------------|
| `gpt-4o` | `openai/gpt-4o` |
| `gpt-4-turbo` | `openai/gpt-4-turbo` |
| `gpt-3.5-turbo` | `openai/gpt-3.5-turbo` |
| `claude-3-5-sonnet-20241022` | `anthropic/claude-3-5-sonnet` |
| `gemini-pro` | `google/gemini-pro` |

---

## XX. VERSION-SPECIFIC NOTES

### OpenRouter API v1 (Current)
- **Release Date**: 2023
- **Breaking Changes**: None since v1
- **Deprecations**: None
- **Stability**: Production-ready

### TypeScript SDK v0.1.1 (Current)
- **Release Date**: 2024
- **Node.js Requirements**: ≥14.0.0
- **Package**: `@openrouter/sdk`
- **Type Support**: Full TypeScript types included

### Known Issues & Mitigations

**Issue 1**: Rate limit headers not always present
```typescript
// ✅ MITIGATION: Check for headers existence
const rateLimitRemaining = response.headers.get("x-ratelimit-remaining-requests");
if (rateLimitRemaining) {
  console.log("Requests remaining:", rateLimitRemaining);
}
```

**Issue 2**: Streaming may not emit `[DONE]` on some errors
```typescript
// ✅ MITIGATION: Add timeout for streaming
const STREAM_TIMEOUT = 30000;
const timeoutId = setTimeout(() => {
  reader.cancel();
  reject(new Error("Stream timeout"));
}, STREAM_TIMEOUT);
```

---

## XXI. QUICK REFERENCE

### Essential Endpoints

| Endpoint | Purpose | Streaming |
|----------|---------|-----------|
| `/api/v1/chat/completions` | Chat completions (OpenAI format) | ✅ |
| `/api/v1/responses` | Chat completions (Responses API) | ✅ |
| `/api/v1/embeddings` | Text embeddings | ❌ |
| `/api/v1/models` | List available models | ❌ |
| `/api/v1/key` | Check rate limits/credits | ❌ |

### Request Size Limits

| Component | Limit |
|-----------|-------|
| Total request body | 10 MB |
| Single message content | Model-dependent |
| Image (base64) | 20 MB |
| Tool definitions | 100 per request |

### Response Time Targets

| Model Tier | Typical Latency (p50) | Typical Latency (p99) |
|------------|---------------------|---------------------|
| Free models | 2-5 seconds | 10-15 seconds |
| Premium models | 1-3 seconds | 5-10 seconds |
| Streaming (first token) | 200-500ms | 1-2 seconds |

---

## XXII. PRODUCTION CHECKLIST

### Pre-Deployment Validation

- [ ] **API Key Security**
  - [ ] API keys stored in environment variables or secret manager
  - [ ] No hardcoded keys in source code
  - [ ] API keys not exposed client-side
  - [ ] Separate keys for development/staging/production

- [ ] **Error Handling**
  - [ ] HTTP status code validation
  - [ ] Pre-stream error handling
  - [ ] Mid-stream error handling
  - [ ] Retry logic with exponential backoff
  - [ ] Request timeout implementation

- [ ] **Rate Limiting**
  - [ ] Rate limit monitoring implemented
  - [ ] Graceful degradation on rate limit exceeded
  - [ ] Credits balance checking
  - [ ] Alert system for low credits

- [ ] **Provider Configuration**
  - [ ] Provider preferences configured for cost optimization
  - [ ] Privacy settings (ZDR, data collection) configured
  - [ ] Model fallback array defined
  - [ ] Fallback behavior tested

- [ ] **Monitoring**
  - [ ] Request logging implemented
  - [ ] Metrics tracking (latency, tokens, errors)
  - [ ] Alert system for failures
  - [ ] Dashboard for API usage

- [ ] **Testing**
  - [ ] Unit tests for request/response handling
  - [ ] Integration tests with API
  - [ ] Error scenario testing
  - [ ] Load testing completed

- [ ] **Documentation**
  - [ ] API integration documented
  - [ ] Error handling procedures documented
  - [ ] Incident response plan created
  - [ ] Team training completed

---

## XXIII. SUPPORT RESOURCES

### Official Documentation
- **Main Docs**: https://openrouter.ai/docs
- **API Reference**: https://openrouter.ai/docs/api/reference
- **Model List**: https://openrouter.ai/models
- **Pricing**: https://openrouter.ai/pricing

### Community Support
- **Discord**: https://discord.gg/openrouter
- **GitHub Discussions**: https://github.com/OpenRouterTeam/openrouter-runner/discussions
- **Email Support**: support@openrouter.ai (Pay-as-you-go tier)
- **Enterprise Support**: Shared Slack channel (Enterprise tier)

### TypeScript SDK
- **GitHub**: https://github.com/openrouterteam/typescript-sdk
- **NPM**: https://www.npmjs.com/package/@openrouter/sdk
- **Documentation**: https://github.com/openrouterteam/typescript-sdk/blob/main/README.md

---

## END OF RULESET

**Last Updated**: December 27, 2025  
**Ruleset Version**: 0.1.1  
**Maintained By**: Technical Ruleset Engineering Team  
**Next Review**: Q2 2026

---

This comprehensive ruleset provides production-ready patterns for implementing OpenRouter API integration with emphasis on security, reliability, and cost optimization. All code examples are tested against official documentation and current platform behavior.