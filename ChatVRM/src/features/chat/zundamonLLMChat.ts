import { Message } from "../messages/messages";

export async function getLocalChatResponseStream(
  text: string,
) {

  const res = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      'text': text,
      'max_new_tokens': 128,
    }),
  })
  const reader = res.body?.getReader();
  if (res.status !== 200 || !reader) {
    throw new Error("Something went wrong");
  }

  const stream = new ReadableStream({
    async start(controller: ReadableStreamDefaultController) {
      const decoder = new TextDecoder("utf-8");
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const data = decoder.decode(value);
          // const chunks = data
          //   .split("data:")
          //   .filter((val) => !!val && val.trim() !== "[DONE]");
          // for (const chunk of chunks) {
          //   const json = JSON.parse(chunk);
          //   const messagePiece = json.choices[0].delta.content;
          //   if (!!messagePiece) {
          //     controller.enqueue(messagePiece);
          //   }
          // }
          controller.enqueue(data);
        }
      } catch (error) {
        controller.error(error);
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });

  return stream;
}
