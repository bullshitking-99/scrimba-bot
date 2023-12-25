export function formatConvHistory(messages) {
  return messages
    .map((message, i) => {
      if (i % 2) {
        return `Human: ${message}`;
      } else {
        return `AI: ${message}`;
      }
    })
    .join("\n");
}
