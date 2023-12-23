import "./App.css";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import info from "./assets/scrimba-info.txt";
import { useEffect } from "react";

function App() {
  async function chunk_split_embedding_store() {
    try {
      const result = await fetch(info);
      const text = await result.text();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
        separators: ["\n\n", "\n", " ", ""], // default setting
      });

      const output = await splitter.createDocuments([text]);

      const sbApiKey =
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imttbm5xemFvYnVjYmx5d3llZW9nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDIyMjA3NjIsImV4cCI6MjAxNzc5Njc2Mn0.VEobY_9tF_oG6IYipLvzI3LjdGBRD4E250nKvjT5OQk";
      const sbUrl = "https://kmnnqzaobucblywyeeog.supabase.co";
      const openAIApiKey =
        "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";

      const client = createClient(sbUrl, sbApiKey);

      const res = await SupabaseVectorStore.fromDocuments(
        output,
        new OpenAIEmbeddings({ openAIApiKey }),
        {
          client,
          tableName: "documents",
        }
      );

      console.log(res);
    } catch (err) {
      console.log("error occur:", err);
    }
  }

  async function setup() {
    // 初始背景文档获取、分块、向量化、存储
    // await chunk_split_embedding_store();
    // 根据input生成standAloneInput
    // 在向量数据库中查找最近解
    // 序列化处理原始input
  }

  useEffect(() => setup, []);

  return <h1>?</h1>;
}

export default App;
