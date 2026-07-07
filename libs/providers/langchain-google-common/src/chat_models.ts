import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { type BaseMessage } from "@langchain/core/messages";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { convertGoogleGeminiStream } from "./utils/stream_events.js";
import { JsonStream } from "./utils/stream.js";

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
import { Runnable } from "@langchain/core/runnables";
import { AsyncCaller } from "@langchain/core/utils/async_caller";
import { concat } from "@langchain/core/utils/stream";
import { v4 as uuidv4 } from "@langchain/core/utils/uuid";
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
import {
  isSerializableSchema,
  SerializableSchema,
} from "@langchain/core/utils/standard_schema";
import {
  assembleStructuredOutputPipeline,
  createContentParser,
  createFunctionCallingParser,
} from "@langchain/core/language_models/structured_output";

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
    this._addVersion("@langchain/google-common", __PKG_VERSION__);

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

  async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    const parameters = this.invocationParams(options);
    const response = await this.streamedConnection.request(
      messages,
      parameters,
      options,
      _runManager
    );
    const stream = response.data as JsonStream;
    const shouldStreamUsage =
      this.streamUsage !== false && options.streamUsage !== false;
    async function* geminiChunks(jsonStream: JsonStream, signal?: AbortSignal) {
      while (!jsonStream.streamDone) {
        if (signal?.aborted) {
          return;
        }
        const output = await jsonStream.nextChunk();
        if (output !== null) {
          yield output;
        }
      }
    }
    yield* convertGoogleGeminiStream(geminiChunks(stream, options.signal), {
      streamUsage: shouldStreamUsage,
    });
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
    const shouldStreamUsage =
      this.streamUsage !== false && options.streamUsage !== false;

    const isStreamingFunctionCalls =
      parameters.streamFunctionCallArguments === true;

    type ActiveToolCall = {
      name: string;
      id: string;
      index: number;
      accumulatedArgs: Record<string, unknown>;
      hasPartialArgs: boolean;
      thoughtSignature?: string;
    };

    let activeTool: ActiveToolCall | undefined;
    let nextToolCallIndex = 0;
    const streamToolCallIdNamespace = uuidv4().replace(/-/g, "");
    const streamToolCallId = (index: number): string =>
      `lc-tool-call-${streamToolCallIdNamespace}-${index}`;

    const streamingSafeAdditionalKwargs = (
      kwargs: AIMessageChunk["additional_kwargs"]
    ): AIMessageChunk["additional_kwargs"] => {
      if (!kwargs) return kwargs;
      const next = { ...kwargs };
      delete next.tool_calls;
      delete next.function_call;
      delete next.signatures;
      return Object.keys(next).length > 0 ? next : {};
    };

    const firstSignature = (
      kwargs: AIMessageChunk["additional_kwargs"]
    ): string | undefined => {
      const signatures = kwargs?.signatures;
      if (!Array.isArray(signatures)) return undefined;
      return signatures.find(
        (signature): signature is string =>
          typeof signature === "string" && signature.length > 0
      );
    };

    const chunkFromMessage = (
      source: ChatGenerationChunk,
      message: AIMessageChunk
    ): ChatGenerationChunk =>
      new ChatGenerationChunk({
        text: source.text,
        generationInfo: source.generationInfo,
        message,
      });

    const emitToolFlush = async (
      tool: ActiveToolCall
    ): Promise<ChatGenerationChunk> => {
      const args = tool.hasPartialArgs
        ? JSON.stringify(tool.accumulatedArgs)
        : "{}";
      const chunk = new ChatGenerationChunk({
        text: "",
        message: new AIMessageChunk({
          content: "",
          additional_kwargs: tool.thoughtSignature
            ? { signatures: [tool.thoughtSignature] }
            : {},
          tool_call_chunks: [
            {
              name: tool.name,
              args,
              id: tool.id,
              index: tool.index,
              type: "tool_call_chunk" as const,
            },
          ],
        }),
      });
      await runManager?.handleLLMNewToken(
        "",
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
      return chunk;
    };

    // Loop until the end of the stream
    // During the loop, yield each time we get a chunk from the streaming parser
    // that is either available or added to the queue
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
      if (output === null) {
        continue;
      }

      let chunk = this.connection.api.responseToChatGeneration({
        data: output,
      });

      if (shouldStreamUsage && chunk) {
        chunk.message = new AIMessageChunk({
          ...chunk.message,
          usage_metadata: chunk.generationInfo?.usage_metadata,
        });
      }

      if (chunk) {
        const msg = chunk.message as AIMessageChunk;

        if (isStreamingFunctionCalls) {
          let startSignalName: string | undefined;
          let hasPartials = false;

          const toolCallChunks = msg.tool_call_chunks ?? [];
          for (const toolCallChunk of toolCallChunks) {
            if (toolCallChunk.args === "{}" && toolCallChunk.name) {
              startSignalName = toolCallChunk.name;
            } else if (
              toolCallChunk.args &&
              toolCallChunk.args !== "{}" &&
              !toolCallChunk.name
            ) {
              hasPartials = true;
            }
          }

          if (startSignalName) {
            if (activeTool) {
              yield await emitToolFlush(activeTool);
            }

            const activeToolIndex = nextToolCallIndex++;
            activeTool = {
              name: startSignalName,
              id: streamToolCallId(activeToolIndex),
              index: activeToolIndex,
              accumulatedArgs: {},
              hasPartialArgs: false,
              thoughtSignature: firstSignature(msg.additional_kwargs),
            };

            chunk = chunkFromMessage(
              chunk,
              new AIMessageChunk({
                content: msg.content,
                additional_kwargs: streamingSafeAdditionalKwargs(
                  msg.additional_kwargs
                ),
                response_metadata: msg.response_metadata,
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
              })
            );
          } else if (!hasPartials) {
            chunk = chunkFromMessage(
              chunk,
              new AIMessageChunk({
                content: msg.content,
                additional_kwargs: streamingSafeAdditionalKwargs(
                  msg.additional_kwargs
                ),
                response_metadata: msg.response_metadata,
                usage_metadata: msg.usage_metadata,
                tool_call_chunks: msg.tool_call_chunks,
              })
            );
          }

          if (hasPartials && activeTool) {
            for (const toolCallChunk of toolCallChunks) {
              if (
                toolCallChunk.args &&
                toolCallChunk.args !== "{}" &&
                !toolCallChunk.name
              ) {
                activeTool.hasPartialArgs = true;
                try {
                  const parsed = JSON.parse(toolCallChunk.args) as Record<
                    string,
                    unknown
                  >;
                  for (const [key, value] of Object.entries(parsed)) {
                    if (
                      typeof value === "string" &&
                      typeof activeTool.accumulatedArgs[key] === "string"
                    ) {
                      activeTool.accumulatedArgs[key] =
                        `${activeTool.accumulatedArgs[key]}${value}`;
                    } else {
                      activeTool.accumulatedArgs[key] = value;
                    }
                  }
                } catch {
                  // Ignore malformed intermediate fragments; the final model
                  // output will still be validated by downstream tool parsing.
                }
              }
            }

            chunk = chunkFromMessage(
              chunk,
              new AIMessageChunk({
                content: msg.content,
                additional_kwargs: streamingSafeAdditionalKwargs(
                  msg.additional_kwargs
                ),
                response_metadata: msg.response_metadata,
                usage_metadata: msg.usage_metadata,
              })
            );
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

    if (isStreamingFunctionCalls && activeTool) {
      yield await emitToolFlush(activeTool);
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
    // oxlint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      // oxlint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<false>
  ): Runnable<BaseLanguageModelInput, RunOutput>;

  withStructuredOutput<
    // oxlint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      // oxlint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<true>
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  withStructuredOutput<
    // oxlint-disable-next-line @typescript-eslint/no-explicit-any
    RunOutput extends Record<string, any> = Record<string, any>,
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      // oxlint-disable-next-line @typescript-eslint/no-explicit-any
      | Record<string, any>,
    config?: StructuredOutputMethodOptions<boolean>
  ):
    | Runnable<BaseLanguageModelInput, RunOutput>
    | Runnable<
        BaseLanguageModelInput,
        { raw: BaseMessage; parsed: RunOutput }
      > {
    const schema = outputSchema;
    const name = config?.name;
    const method = config?.method;
    const includeRaw = config?.includeRaw;

    if (method === "jsonMode") {
      throw new Error(
        `Google only supports "jsonSchema" or "functionCalling" as a method.`
      );
    }

    let llm: Runnable<BaseLanguageModelInput>;
    let outputParser: Runnable<AIMessageChunk, RunOutput>;

    if (method === "functionCalling") {
      let functionName = name ?? "extract";
      let geminiFunctionDeclaration: GeminiFunctionDeclaration;
      if (isInteropZodSchema(schema) || isSerializableSchema(schema)) {
        const jsonSchema = schemaToGeminiParameters(schema);
        geminiFunctionDeclaration = {
          name: functionName,
          description:
            jsonSchema.description ?? "A function available to call.",
          parameters: jsonSchema as GeminiFunctionSchema,
        };
      } else if (
        typeof schema.name === "string" &&
        typeof schema.parameters === "object" &&
        schema.parameters != null
      ) {
        geminiFunctionDeclaration = schema as GeminiFunctionDeclaration;
        functionName = schema.name;
      } else {
        // We are providing the schema for *just* the parameters, probably
        const parameters: GeminiJsonSchema = removeAdditionalProperties(schema);
        geminiFunctionDeclaration = {
          name: functionName,
          description: schema.description ?? "",
          parameters,
        };
      }

      const tools: GeminiTool[] = [
        { functionDeclarations: [geminiFunctionDeclaration] },
      ];
      llm = this.bindTools(tools).withConfig({ tool_choice: functionName });

      outputParser = createFunctionCallingParser(schema, functionName);
    } else {
      // Default to jsonSchema method
      const jsonSchema = schemaToGeminiParameters(schema);
      llm = this.withConfig({
        responseSchema: jsonSchema as GeminiJsonSchema,
      });
      outputParser = createContentParser(schema);
    }

    return assembleStructuredOutputPipeline(
      llm,
      outputParser,
      includeRaw,
      includeRaw ? "StructuredOutputRunnable" : "ChatGoogleStructuredOutput"
    );
  }
}
