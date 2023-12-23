import "./App.css";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "langchain/schema/output_parser";

import info from "./assets/scrimba-info.txt";
import { useEffect } from "react";
import { retriever, embeddings, client, llm } from "./utils/common";
import { combineDocuments } from "./utils/combineDocuments";

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

      const res = await SupabaseVectorStore.fromDocuments(output, embeddings, {
        client,
        tableName: "documents",
      });

      // console.log(res);
    } catch (err) {
      console.log("error occur:", err);
    }
  }

  async function generate_standAloneInput_retrievel() {
    const standaloneQuestionTemplate =
      "Given a question, convert it to a standalone question. question: {question} standalone question:";
    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
      standaloneQuestionTemplate
    );

    const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer: 
`;
    const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

    const chain = standaloneQuestionPrompt
      .pipe(llm)
      .pipe(new StringOutputParser())
      .pipe(retriever)
      .pipe(combineDocuments);
    // .pipe(answerPrompt);

    const response = await chain.invoke({
      question:
        "What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.",
    });

    console.log(response);
  }

  async function setup() {
    // 初始背景文档获取、分块、向量化、存储
    // await chunk_split_embedding_store();

    // 根据input生成standAloneInput
    await generate_standAloneInput_retrievel();

    // 在向量数据库中查找最近解
    // 序列化处理原始input
  }

  useEffect(() => setup, []);

  return <h1>?</h1>;
}

export default App;
