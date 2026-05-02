/**
 * Thrown when converting LangChain message content to Google Gemini parts encounters a block type
 * that cannot be represented on the Google provider (often foreign provider-native shapes replayed from checkpoints).
 */
export class CrossProviderContentError extends Error {
  public readonly code = "CROSS_PROVIDER_CONTENT" as const;

  public constructor(public readonly providerNativeType: string) {
    super(
      `Content block of type "${providerNativeType}" is not portable to the Google provider. ` +
        `This usually means assistant history from another provider was replayed without normalization. ` +
        `Switch back to the originating model or normalize via standard V1 content blocks.`
    );
    this.name = "CrossProviderContentError";
  }
}
