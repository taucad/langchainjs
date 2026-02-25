import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { UsageMetadata, type BaseMessage } from "@langchain/core/messages";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

import {
  BaseChatModel,
  LangSmithParams,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import { ChatGenerationChunk, ChatResult } from "@langchain/core/outputs";
import { AIMessageChunk } from "@langchain/core/messages";
import {
  BaseLanguageModelInput,
  StructuredOutputMethodOptions,
} from "@langchain/core/language_models/base";
import { type ModelProfile } from "@langchain/core/language_models/profile";
import {
  Runnable,
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { JsonOutputKeyToolsParser } from "@langchain/core/output_parsers/openai_tools";
import {
  BaseLLMOutputParser,
  JsonOutputParser,
} from "@langchain/core/output_parsers";
import { AsyncCaller } from "@langchain/core/utils/async_caller";
import { concat } from "@langchain/core/utils/stream";
import {
  InteropZodType,
  isInteropZodSchema,
} from "@langchain/core/utils/types";
import {
  GoogleAIBaseLLMInput,
  GoogleAIModelParams,
  GoogleAISafetySetting,
  GoogleConnectionParams,
  GooglePlatformType,
  GeminiTool,
  GoogleAIBaseLanguageModelCallOptions,
  GoogleAIAPI,
  GoogleAIAPIParams,
  GoogleSearchToolSetting,
  GoogleSpeechConfig,
  GeminiJsonSchema,
} from "./types.js";
import {
  convertToGeminiTools,
  copyAIModelParams,
  copyAndValidateModelParamsInto,
} from "./utils/common.js";
import { AbstractGoogleLLMConnection } from "./connection.js";
import { DefaultGeminiSafetyHandler, getGeminiAPI } from "./utils/gemini.js";
import { ApiKeyGoogleAuth, GoogleAbstractedClient } from "./auth.js";
import { JsonStream } from "./utils/stream.js";
import { ensureParams } from "./utils/failed_handler.js";
import type {
  GoogleBaseLLMInput,
  GoogleAISafetyHandler,
  GoogleAISafetyParams,
  GeminiFunctionDeclaration,
  GeminiFunctionSchema,
  GoogleAIToolType,
  GeminiAPIConfig,
  GoogleAIModelModality,
} from "./types.js";
import {
  removeAdditionalProperties,
  schemaToGeminiParameters,
} from "./utils/zod_to_gemini_parameters.js";
import PROFILES from "./profiles.js";

export class ChatConnection<AuthOptions> extends AbstractGoogleLLMConnection<
  BaseMessage[],
  AuthOptions
> {
  convertSystemMessageToHumanContent: boolean | undefined;

  constructor(
    fields: GoogleAIBaseLLMInput<AuthOptions> | undefined,
    caller: AsyncCaller,
    client: GoogleAbstractedClient,
    streaming: boolean
  ) {
    super(fields, caller, client, streaming);
    this.convertSystemMessageToHumanContent =
      fields?.convertSystemMessageToHumanContent;
  }

  get useSystemInstruction(): boolean {
    return typeof this.convertSystemMessageToHumanContent === "boolean"
      ? !this.convertSystemMessageToHumanContent
      : this.computeUseSystemInstruction;
  }

  get computeUseSystemInstruction(): boolean {
    // This works on models from April 2024 and later
    //   Vertex AI: gemini-1.5-pro and gemini-1.0-002 and later
    //   AI Studio: gemini-1.5-pro-latest
    if (this.modelFamily === "palm") {
      return false;
    } else if (this.modelName === "gemini-1.0-pro-001") {
      return false;
    } else if (this.modelName.startsWith("gemini-pro-vision")) {
      return false;
    } else if (this.modelName.startsWith("gemini-1.0-pro-vision")) {
      return false;
    } else if (this.modelName === "gemini-pro" && this.platform === "gai") {
      // on AI Studio gemini-pro is still pointing at gemini-1.0-pro-001
      return false;
    } else if (this.modelFamily === "gemma") {
      // At least as of 12 Mar 2025 gemma 3 on AIS, trying to use system instructions yields an error:
      // "Developer instruction is not enabled for models/gemma-3-27b-it"
      return false;
    }
    return true;
  }

  computeGoogleSearchToolAdjustmentFromModel(): Exclude<
    GoogleSearchToolSetting,
    boolean
  > {
    if (this.modelName.startsWith("gemini-1.0")) {
      return "googleSearchRetrieval";
    } else if (this.modelName.startsWith("gemini-1.5")) {
      return "googleSearchRetrieval";
    } else {
      return "googleSearch";
    }
  }

  computeGoogleSearchToolAdjustment(
    apiConfig: GeminiAPIConfig
  ): Exclude<GoogleSearchToolSetting, true> {
    const adj = apiConfig.googleSearchToolAdjustment;
    if (adj === undefined || adj === true) {
      return this.computeGoogleSearchToolAdjustmentFromModel();
    } else {
      return adj;
    }
  }

  buildGeminiAPI(): GoogleAIAPI {
    const apiConfig: GeminiAPIConfig =
      (this.apiConfig as GeminiAPIConfig) ?? {};
    const googleSearchToolAdjustment =
      this.computeGoogleSearchToolAdjustment(apiConfig);
    const geminiConfig: GeminiAPIConfig = {
      useSystemInstruction: this.useSystemInstruction,
      googleSearchToolAdjustment,
      ...apiConfig,
    };
    return getGeminiAPI(geminiConfig);
  }

  get api(): GoogleAIAPI {
    switch (this.apiName) {
      case "google":
        return this.buildGeminiAPI();
      default:
        return super.api;
    }
  }
}

/**
 * Input to chat model class.
 */
export interface ChatGoogleBaseInput<AuthOptions>
  extends
    BaseChatModelParams,
    GoogleConnectionParams<AuthOptions>,
    GoogleAIModelParams,
    GoogleAISafetyParams,
    GoogleAIAPIParams,
    Pick<GoogleAIBaseLanguageModelCallOptions, "streamUsage"> {}

/**
 * Integration with a Google chat model.
 */
export abstract class ChatGoogleBase<AuthOptions>
  extends BaseChatModel<GoogleAIBaseLanguageModelCallOptions, AIMessageChunk>
  implements ChatGoogleBaseInput<AuthOptions>
{
  // Used for tracing, replace with the same name as your class
  static lc_name() {
    return "ChatGoogle";
  }

  get lc_secrets(): { [key: string]: string } | undefined {
    return {
      authOptions: "GOOGLE_AUTH_OPTIONS",
    };
  }

  lc_serializable = true;

  // Set based on modelName
  model: string;

  modelName = "gemini-pro";

  temperature: number;

  maxOutputTokens: number;

  maxReasoningTokens: number;

  topP: number;

  topK: number;

  seed: number;

  presencePenalty: number;

  frequencyPenalty: number;

  stopSequences: string[] = [];

  logprobs: boolean;

  topLogprobs: number = 0;

  safetySettings: GoogleAISafetySetting[] = [];

  responseModalities?: GoogleAIModelModality[];

  // May intentionally be undefined, meaning to compute this.
  convertSystemMessageToHumanContent: boolean | undefined;

  safetyHandler: GoogleAISafetyHandler;

  speechConfig: GoogleSpeechConfig;

  streamUsage = true;

  streaming = false;

  labels?: Record<string, string>;

  protected connection: ChatConnection<AuthOptions>;

  protected streamedConnection: ChatConnection<AuthOptions>;

  constructor(fields?: ChatGoogleBaseInput<AuthOptions>) {
    super(ensureParams(fields));

    copyAndValidateModelParamsInto(fields, this);
    this.safetyHandler =
      fields?.safetyHandler ?? new DefaultGeminiSafetyHandler();
    this.streamUsage = fields?.streamUsage ?? this.streamUsage;
    const client = this.buildClient(fields);
    this.buildConnection(fields ?? {}, client);
  }

  getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    const params = this.invocationParams(options);
    return {
      ls_provider: "google_vertexai",
      ls_model_name: this.model,
      ls_model_type: "chat",
      ls_temperature: params.temperature ?? undefined,
      ls_max_tokens: params.maxOutputTokens ?? undefined,
      ls_stop: options.stop,
    };
  }

  abstract buildAbstractedClient(
    fields?: GoogleAIBaseLLMInput<AuthOptions>
  ): GoogleAbstractedClient;

  buildApiKeyClient(apiKey: string): GoogleAbstractedClient {
    return new ApiKeyGoogleAuth(apiKey);
  }

  buildApiKey(fields?: GoogleAIBaseLLMInput<AuthOptions>): string | undefined {
    return fields?.apiKey ?? getEnvironmentVariable("GOOGLE_API_KEY");
  }

  buildClient(
    fields?: GoogleAIBaseLLMInput<AuthOptions>
  ): GoogleAbstractedClient {
    const apiKey = this.buildApiKey(fields);
    if (apiKey) {
      return this.buildApiKeyClient(apiKey);
    } else {
      return this.buildAbstractedClient(fields);
    }
  }

  buildConnection(
    fields: GoogleBaseLLMInput<AuthOptions>,
    client: GoogleAbstractedClient
  ) {
    this.connection = new ChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      false
    );

    this.streamedConnection = new ChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      true
    );
  }

  get platform(): GooglePlatformType {
    return this.connection.platform;
  }

  override bindTools(
    tools: GoogleAIToolType[],
    kwargs?: Partial<GoogleAIBaseLanguageModelCallOptions>
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    GoogleAIBaseLanguageModelCallOptions
  > {
    return this.withConfig({ tools: convertToGeminiTools(tools), ...kwargs });
  }

  _llmType() {
    return "google";
  }

  /**
   * Get the parameters used to invoke the model
   */
  override invocationParams(options?: this["ParsedCallOptions"]) {
    return copyAIModelParams(this, options);
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager: CallbackManagerForLLMRun | undefined
  ): Promise<ChatResult> {
    options.signal?.throwIfAborted();
    const parameters = this.invocationParams(options);
    if (this.streaming) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      let finalChunk: ChatGenerationChunk | null = null;
      for await (const chunk of stream) {
        finalChunk = !finalChunk ? chunk : concat(finalChunk, chunk);
      }
      if (!finalChunk) {
        throw new Error("No chunks were returned from the stream.");
      }
      return {
        generations: [finalChunk],
      };
    }

    const response = await this.connection.request(
      messages,
      parameters,
      options,
      runManager
    );
    const ret = this.connection.api.responseToChatResult(response);
    const chunk = ret?.generations?.[0];
    if (chunk) {
      await runManager?.handleLLMNewToken(chunk.text || "");
    }
    return ret;
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    // Make the call as a streaming request
    const parameters = this.invocationParams(options);
    const response = await this.streamedConnection.request(
      _messages,
      parameters,
      options,
      runManager
    );

    // Get the streaming parser of the response
    const stream = response.data as JsonStream;
    let usageMetadata: UsageMetadata | undefined;

    // When streamFunctionCallArguments is enabled, the Gemini API sends
    // per tool call:
    //  1. A functionCall with name + empty args {} (start signal)
    //  2. Multiple partialArgs chunks with actual argument data
    //  3. A functionCall with empty args {} and no name (end signal)
    //
    // For parallel tool calls, signals repeat sequentially:
    //   start(tool1) → partials(tool1) → end(tool1) →
    //   start(tool2) → partials(tool2) → end(tool2)
    //
    // Unlike OpenAI/Anthropic, Gemini's partial args are complete JSON
    // objects (not concatenable substrings) and the index resets to 0 for
    // each tool call. We must accumulate per tool call and assign a
    // provider-side incrementing index so LangChain core's _mergeLists /
    // collapseToolCallChunks can group chunks correctly.
    const isStreamingFunctionCalls =
      parameters.streamFunctionCallArguments === true;

    type ActiveToolCall = {
      name: string;
      id: string;
      index: number;
      accumulatedArgs: Record<string, unknown>;
      hasPartialArgs: boolean;
    };

    let activeTool: ActiveToolCall | undefined;
    let nextToolCallIndex = 0;

    // Loop until the end of the stream
    while (!stream.streamDone) {
      if (options.signal?.aborted) {
        return;
      }
      const output = await stream.nextChunk();
      await runManager?.handleCustomEvent(
        `google-chunk-${this.constructor.name}`,
        {
          output,
        }
      );
      if (
        output &&
        output.usageMetadata &&
        this.streamUsage !== false &&
        options.streamUsage !== false
      ) {
        usageMetadata = {
          input_tokens: output.usageMetadata.promptTokenCount,
          output_tokens: output.usageMetadata.candidatesTokenCount,
          total_tokens: output.usageMetadata.totalTokenCount,
        };
      }
      let chunk =
        output !== null
          ? this.connection.api.responseToChatGeneration({ data: output })
          : new ChatGenerationChunk({
              text: "",
              generationInfo: { finishReason: "stop" },
              message: new AIMessageChunk({
                content: "",
                usage_metadata: usageMetadata,
              }),
            });
      if (chunk) {
        const msg = chunk.message as AIMessageChunk;

        if (isStreamingFunctionCalls) {
          // Classify this chunk's tool_call_chunks.
          let startSignalName: string | undefined;
          let startSignalId: string | undefined;
          let hasEndSignal = false;
          let hasPartials = false;

          const tccList = msg.tool_call_chunks ?? [];

          for (const tcc of tccList) {
            if (tcc.args === "{}" && tcc.name) {
              // Start signal: has a function name + empty args
              startSignalName = tcc.name;
              startSignalId = tcc.id ?? "";
            } else if (tcc.args === "{}" && !tcc.name) {
              // End signal: empty args, no function name
              hasEndSignal = true;
            } else if (tcc.args && tcc.args !== "{}") {
              // Partial args: non-empty, non-placeholder args
              hasPartials = true;
            }
          }

          // --- Start signal: flush previous tool, create new one ---
          if (startSignalName) {
            // Flush the previous activeTool inline if it has accumulated args.
            if (activeTool?.hasPartialArgs) {
              const syntheticId =
                activeTool.id || `call_${activeTool.name}`;
              const flushChunk = new ChatGenerationChunk({
                text: "",
                message: new AIMessageChunk({
                  content: "",
                  tool_call_chunks: [
                    {
                      name: activeTool.name,
                      args: JSON.stringify(activeTool.accumulatedArgs),
                      id: syntheticId,
                      index: activeTool.index,
                      type: "tool_call_chunk" as const,
                    },
                  ],
                }),
              });
              yield flushChunk;
              await runManager?.handleLLMNewToken(
                "",
                undefined,
                undefined,
                undefined,
                undefined,
                { chunk: flushChunk }
              );
            }

            // Create new activeTool with an incrementing index.
            activeTool = {
              name: startSignalName,
              id: startSignalId ?? "",
              index: nextToolCallIndex++,
              accumulatedArgs: {},
              hasPartialArgs: false,
            };

            // Emit start chunk with args: "" so the adapter emits
            // tool-input-start. collapseToolCallChunks merges by index.
            chunk = new ChatGenerationChunk({
              text: chunk.text,
              generationInfo: chunk.generationInfo,
              message: new AIMessageChunk({
                content: msg.content,
                additional_kwargs: msg.additional_kwargs,
                usage_metadata: msg.usage_metadata,
                tool_call_chunks: [
                  {
                    name: activeTool.name,
                    args: "",
                    id: activeTool.id,
                    index: activeTool.index,
                    type: "tool_call_chunk" as const,
                  },
                ],
              }),
            });
          }

          // --- Accumulate partial args into the active tool ---
          if (hasPartials && activeTool) {
            for (const tcc of msg.tool_call_chunks ?? []) {
              if (tcc.args && tcc.args !== "{}") {
                activeTool.hasPartialArgs = true;
                try {
                  const parsed = JSON.parse(tcc.args) as Record<
                    string,
                    unknown
                  >;
                  for (const [k, v] of Object.entries(parsed)) {
                    if (
                      typeof v === "string" &&
                      typeof activeTool.accumulatedArgs[k] === "string"
                    ) {
                      (activeTool.accumulatedArgs[k] as string) += v as string;
                    } else {
                      activeTool.accumulatedArgs[k] = v;
                    }
                  }
                } catch {
                  // Fragment is not valid JSON; skip accumulation
                }
              }
            }

            // Suppress tool_call_chunks from partial arg chunks so they
            // don't get concatenated by _mergeLists and produce invalid JSON.
            // Only the start and flush chunks should carry tool_call_chunks.
            chunk = new ChatGenerationChunk({
              text: chunk.text,
              generationInfo: chunk.generationInfo,
              message: new AIMessageChunk({
                content: msg.content,
                additional_kwargs: msg.additional_kwargs,
                usage_metadata: msg.usage_metadata,
              }),
            });
          }

          // --- End signal: suppress tool_call_chunks ---
          if (hasEndSignal && !startSignalName) {
            chunk = new ChatGenerationChunk({
              text: chunk.text,
              generationInfo: chunk.generationInfo,
              message: new AIMessageChunk({
                content: msg.content,
                additional_kwargs: msg.additional_kwargs,
                usage_metadata: msg.usage_metadata,
              }),
            });
          }
        }

        yield chunk;
        await runManager?.handleLLMNewToken(
          chunk.text ?? "",
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk }
        );
      }
    }

    // Flush the last activeTool after the stream ends.
    if (isStreamingFunctionCalls && activeTool) {
      if (activeTool.hasPartialArgs) {
        const syntheticId =
          activeTool.id || `call_${activeTool.name}`;
        const syntheticChunk = new ChatGenerationChunk({
          text: "",
          message: new AIMessageChunk({
            content: "",
            tool_call_chunks: [
              {
                name: activeTool.name,
                args: JSON.stringify(activeTool.accumulatedArgs),
                id: syntheticId,
                index: activeTool.index,
                type: "tool_call_chunk" as const,
              },
            ],
          }),
        });
        yield syntheticChunk;
        await runManager?.handleLLMNewToken(
          "",
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: syntheticChunk }
        );
      } else {
        // No partial args arrived — legitimate no-args tool call.
        const noArgChunk = new ChatGenerationChunk({
          text: "",
          message: new AIMessageChunk({
            content: "",
            tool_call_chunks: [
              {
                name: activeTool.name,
                args: "{}",
                id: activeTool.id,
                index: activeTool.index,
                type: "tool_call_chunk" as const,
              },
            ],
          }),
        });
        yield noArgChunk;
        await runManager?.handleLLMNewToken(
          "",
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: noArgChunk }
        );
      }
    }
  }

  /** @ignore */
  _combineLLMOutput() {
    return [];
  }

  /**
   * Return profiling information for the model.
   *
   * Provides information about the model's capabilities and constraints,
   * including token limits, multimodal support, and advanced features like
   * tool calling and structured output.
   *
   * @returns {ModelProfile} An object describing the model's capabilities and constraints
   */
  get profile(): ModelProfile {
    return PROFILES[this.model] ?? {};
  }

  withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<false>
  ): Runnable<BaseLanguageModelInput, RunOutput>;

  withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<true>
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  withStructuredOutput<
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<boolean>
  ):
    | Runnable<BaseLanguageModelInput, RunOutput>
    | Runnable<
        BaseLanguageModelInput,
        { raw: BaseMessage; parsed: RunOutput }
      > {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const schema: InteropZodType<RunOutput> | Record<string, any> =
      outputSchema;
    const name = config?.name;
    const method = config?.method;
    const includeRaw = config?.includeRaw;
    if (method === "jsonMode") {
      throw new Error(
        `Google only supports "jsonSchema" or "functionCalling" as a method.`
      );
    }

    let llm;
    let outputParser: BaseLLMOutputParser<RunOutput>;
    if (method === "functionCalling") {
      let functionName = name ?? "extract";
      let tools: GeminiTool[];
      if (isInteropZodSchema(schema)) {
        const jsonSchema = schemaToGeminiParameters(schema);
        tools = [
          {
            functionDeclarations: [
              {
                name: functionName,
                description:
                  jsonSchema.description ?? "A function available to call.",
                parameters: jsonSchema as GeminiFunctionSchema,
              },
            ],
          },
        ];
        outputParser = new JsonOutputKeyToolsParser({
          returnSingle: true,
          keyName: functionName,
          zodSchema: schema,
        });
      } else {
        let geminiFunctionDefinition: GeminiFunctionDeclaration;
        if (
          typeof schema.name === "string" &&
          typeof schema.parameters === "object" &&
          schema.parameters != null
        ) {
          geminiFunctionDefinition = schema as GeminiFunctionDeclaration;
          functionName = schema.name;
        } else {
          // We are providing the schema for *just* the parameters, probably
          const parameters: GeminiJsonSchema =
            removeAdditionalProperties(schema);
          geminiFunctionDefinition = {
            name: functionName,
            description: schema.description ?? "",
            parameters,
          };
        }
        tools = [
          {
            functionDeclarations: [geminiFunctionDefinition],
          },
        ];
        outputParser = new JsonOutputKeyToolsParser<RunOutput>({
          returnSingle: true,
          keyName: functionName,
        });
      }
      llm = this.bindTools(tools).withConfig({ tool_choice: functionName });
    } else {
      // Default to jsonSchema method
      const jsonSchema = schemaToGeminiParameters(schema);
      llm = this.withConfig({
        responseSchema: jsonSchema as GeminiJsonSchema,
      });
      outputParser = new JsonOutputParser();
    }

    if (!includeRaw) {
      return llm.pipe(outputParser).withConfig({
        runName: "ChatGoogleStructuredOutput",
      }) as Runnable<BaseLanguageModelInput, RunOutput>;
    }

    const parserAssign = RunnablePassthrough.assign({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parsed: (input: any, config) => outputParser.invoke(input.raw, config),
    });
    const parserNone = RunnablePassthrough.assign({
      parsed: () => null,
    });
    const parsedWithFallback = parserAssign.withFallbacks({
      fallbacks: [parserNone],
    });
    return RunnableSequence.from<
      BaseLanguageModelInput,
      { raw: BaseMessage; parsed: RunOutput }
    >([
      {
        raw: llm,
      },
      parsedWithFallback,
    ]).withConfig({
      runName: "StructuredOutputRunnable",
    });
  }
}
