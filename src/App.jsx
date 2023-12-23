import "./App.css";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import info from "./assets/scrimba-info.txt";
import { useEffect } from "react";

import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";

const {
  VITE_sbApiKey: sbApiKey,
  VITE_sbUrl: sbUrl,
  VITE_openAIApiKey: openAIApiKey,
} = import.meta.env;

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

      const client = createClient(sbUrl, sbApiKey);

      const res = await SupabaseVectorStore.fromDocuments(
        output,
        new OpenAIEmbeddings({ openAIApiKey }),
        {
          client,
          tableName: "documents",
        }
      );

      // console.log(res);
    } catch (err) {
      console.log("error occur:", err);
    }
  }

  async function generate_standAlone_Input() {
    const llm = new ChatOpenAI({ openAIApiKey });
    const standaloneQuestionTemplate =
      "Given a question, convert it to a standalone question. question: {question} standalone question:";
    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
      standaloneQuestionTemplate
    );
    const standaloneQuestionChain = standaloneQuestionPrompt.pipe(llm);
    const response = await standaloneQuestionChain.invoke({
      question:
        "What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.",
    });
    console.log(response);
  }

  async function retrieval() {}

  async function setup() {
    // 初始背景文档获取、分块、向量化、存储
    // await chunk_split_embedding_store();

    // 根据input生成standAloneInput
    await generate_standAlone_Input();

    // 在向量数据库中查找最近解
    // 序列化处理原始input
  }

  useEffect(() => setup, []);

  return <h1>?</h1>;
}

export default App;
