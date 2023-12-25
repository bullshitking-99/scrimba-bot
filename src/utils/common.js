import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI } from "langchain/chat_models/openai";

const {
  VITE_sbApiKey: sbApiKey,
  VITE_sbUrl: sbUrl,
  VITE_openAIApiKey: openAIApiKey,
} = import.meta.env;

const llm = new ChatOpenAI({ openAIApiKey });

const embeddings = new OpenAIEmbeddings({ openAIApiKey });
const client = createClient(sbUrl, sbApiKey);

const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: "documents",
  queryName: "match_documents",
});

const retriever = vectorStore.asRetriever();

export { retriever, client, embeddings, llm };
