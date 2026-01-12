package mcp

import (
	"net/http"
)

const (
	ProviderOllama       = "ollama"
	DefaultOllamaBaseURL = "https://ollama.com"
	DefaultOllamaModel   = "glm-4.7:cloud"
)

type OllamaClient struct {
	*Client
}

// NewOllamaClientWithOptions creates Ollama cloud client
//
// Usage examples:
//   // Basic usage
//   client := mcp.NewOllamaClientWithOptions()
//
//   // Custom configuration
//   client := mcp.NewOllamaClientWithOptions(
//       mcp.WithAPIKey("sk-xxx"),
//       mcp.WithLogger(customLogger),
//       mcp.WithTimeout(60*time.Second),
//   )
func NewOllamaClientWithOptions(opts ...ClientOption) AIClient {
	// 1. Create Ollama preset options
	ollamaOpts := []ClientOption{
		WithProvider(ProviderOllama),
		WithModel(DefaultOllamaModel),
		WithBaseURL(DefaultOllamaBaseURL),
	}

	// 2. Merge user options (user options have higher priority)
	allOpts := append(ollamaOpts, opts...)

	// 3. Create base client
	baseClient := NewClient(allOpts...).(*Client)

	// 4. Create Ollama client
	ollamaClient := &OllamaClient{
		Client: baseClient,
	}

	// 5. Set hooks to point to OllamaClient (implement dynamic dispatch)
	baseClient.hooks = ollamaClient

	return ollamaClient
}

func (oc *OllamaClient) SetAPIKey(apiKey string, customURL string, customModel string) {
	oc.APIKey = apiKey

	if len(apiKey) > 8 {
		oc.logger.Infof("🔧 [MCP] Ollama API Key: %s...%s", apiKey[:4], apiKey[len(apiKey)-4:])
	}
	if customURL != "" {
		oc.BaseURL = customURL
		oc.logger.Infof("🔧 [MCP] Ollama using custom BaseURL: %s", customURL)
	} else {
		oc.logger.Infof("🔧 [MCP] Ollama using default BaseURL: %s", oc.BaseURL)
	}
	if customModel != "" {
		oc.Model = customModel
		oc.logger.Infof("🔧 [MCP] Ollama using custom Model: %s", customModel)
	} else {
		oc.logger.Infof("🔧 [MCP] Ollama using default Model: %s", oc.Model)
	}
}

func (oc *OllamaClient) setAuthHeader(reqHeaders http.Header) {
	// Ollama uses Bearer token authentication
	reqHeaders.Set("Authorization", "Bearer "+oc.APIKey)
}

func (oc *OllamaClient) buildUrl() string {
	// Ollama cloud API uses /api/chat endpoint
	return oc.BaseURL + "/api/chat"
}
