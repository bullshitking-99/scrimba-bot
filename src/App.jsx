import "./App.css";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "langchain/schema/output_parser";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "langchain/schema/runnable";

import info from "./assets/scrimba-info.txt";
import { useEffect, useState, useRef } from "react";
import { retriever, embeddings, client, llm } from "./utils/common";
import { combineDocuments } from "./utils/combineDocuments";
import Loading from "./components/loading";
import Send from "./components/send";

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

  async function standalone_retrievel_answer(input) {
    const standaloneQuestionTemplate =
      "Given a question, convert it to a standalone question. question: {question} standalone question:";
    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
      standaloneQuestionTemplate
    );

    const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend. translate the answer to Chinese.
context: {context}
question: {question}
answer: 
`;
    const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

    const standaloneQuestionChain = standaloneQuestionPrompt
      .pipe(llm)
      .pipe(new StringOutputParser());

    const retrieverChain = RunnableSequence.from([
      ({ standalone_question }) => standalone_question,
      retriever,
      combineDocuments,
    ]);

    const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

    const chain = RunnableSequence.from([
      {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough(),
      },
      {
        context: retrieverChain,
        question: ({ original_input }) => original_input.question,
      },
      answerChain,
    ]);

    const response = await chain.invoke({
      question: input,
    });

    return response;
  }

  async function punctuation_grammar_translate(input) {
    const punctuationTemplate = `Given a sentence, add punctuation where needed. 
    sentence: {sentence}
    sentence with punctuation:  
    `;
    const punctuationPrompt = PromptTemplate.fromTemplate(punctuationTemplate);

    const grammarTemplate = `Given a sentence correct the grammar.
    sentence: {punctuated_sentence}
    sentence with correct grammar: 
    `;
    const grammarPrompt = PromptTemplate.fromTemplate(grammarTemplate);

    const translationTemplate = `Given a sentence, translate that sentence into {language}
    sentence: {grammatically_correct_sentence}
    translated sentence:
    `;
    const translationPrompt = PromptTemplate.fromTemplate(translationTemplate);

    const punctuationChain = RunnableSequence.from([
      punctuationPrompt,
      llm,
      new StringOutputParser(),
    ]);
    const grammarChain = RunnableSequence.from([
      grammarPrompt,
      llm,
      new StringOutputParser(),
    ]);
    const translationChain = RunnableSequence.from([
      translationPrompt,
      llm,
      new StringOutputParser(),
    ]);

    const chain = RunnableSequence.from([
      {
        punctuated_sentence: punctuationChain,
        original_input: new RunnablePassthrough(),
      },
      {
        grammatically_correct_sentence: grammarChain,
        language: ({ original_input }) => original_input.language,
      },
      translationChain,
    ]);

    const response = await chain.invoke({
      sentence: input,
      language: "English",
    });

    return response;
  }

  async function setup() {
    // 初始背景文档获取、分块、向量化、存储
    // await chunk_split_embedding_store();
    // 根据input生成standAloneInput，在向量数据库中查找最近解
    // await standalone_retrievel_answer();
    // 序列化纠正原始input
    // await punctuation_grammar_translate(input);
  }

  useEffect(() => setup, []);

  // 对话记录
  const [log, setLog] = useState(["你好，我是Scrimba，有什么能够帮你？"]);

  const addLog = (n) => setLog((old) => [...old, n]);

  const [inputValue, setInputValue] = useState("");

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const [loading, setLoading] = useState(false);

  const chatbotConversationRef = useRef(null);
  const scrollToBottom = () => {
    if (chatbotConversationRef.current) {
      chatbotConversationRef.current.scrollTop =
        chatbotConversationRef.current.scrollHeight;
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault(); // 阻止默认提交行为
    addLog(inputValue);
    const _inputValue = inputValue;
    setInputValue("");
    try {
      setLoading(true);
      // const input = await punctuation_grammar_translate(_inputValue);
      // console.log(input);
      const res = await standalone_retrievel_answer(_inputValue);
      addLog(res);
    } catch (error) {
      addLog("抱歉，好像发生了一些未知错误，请检查一些网络环境或稍后再试。");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // 在组件挂载时或者其他时机执行滚动到底部的逻辑
    scrollToBottom();
  }, [log]); // 这里空数组表示只在组件挂载时执行一次

  return (
    <main>
      <section className="chatbot-container">
        <div className="chatbot-header">
          {/* <img src="./assets/logo-scrimba.svg" className="logo" /> */}
          <p className="sub-heading">Knowledge Bank</p>
        </div>
        <div
          className="chatbot-conversation-container"
          id="chatbot-conversation-container"
          ref={chatbotConversationRef}
        >
          {log.map((l, i) => (
            <div
              key={l + i}
              className={`speech ${i % 2 ? "speech-human" : "speech-ai"}`}
            >
              {l}
            </div>
          ))}
          {loading && <Loading />}
        </div>
        <form
          id="form"
          className="chatbot-input-container"
          onSubmit={handleSubmit}
        >
          <input
            name="user-input"
            type="text"
            id="user-input"
            required
            value={inputValue}
            onChange={handleInputChange}
          />
          <button id="submit-btn" className="submit-btn">
            <Send />
          </button>
        </form>
      </section>
    </main>
  );
}

export default App;
