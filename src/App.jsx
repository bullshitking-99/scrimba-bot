import "./App.css";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "langchain/schema/output_parser";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "langchain/schema/runnable";

import info from "./assets/liminghe-info.txt";
import { useEffect, useState, useRef, useCallback } from "react";
import { retriever, embeddings, client, llm } from "./utils/common";
import { combineDocuments } from "./utils/combineDocuments";
import { formatConvHistory } from "./utils/formatConvHistory";
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

      console.log(res);
    } catch (err) {
      console.log("error occur:", err);
    }
  }

  async function standalone_retrievel_answer(input) {
    const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
    conversation history: {conv_history}
    question: {question} 
    standalone question:`;
    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
      standaloneQuestionTemplate
    );

    const answerTemplate = `你是李名河的数字分身，而非机器人。
    你的目标是：在任何技术、项目管理或职业发展等相关问题上，像李名河本人一样给出专业、亲和、风趣幽默有深度的回答，并在合适的时机加入他个人的思考角度与经验总结。
    你可以结合知识库中的内容自由发挥，但要保持表达风格与李名河一致，语言自然、不生硬，既可以高密度输出技术信息，也能根据上下文放缓节奏、引导用户进一步提问。
    context: {context}
    conversation history: {conv_history}
    question: {question}
    answer: `;
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
        conv_history: ({ original_input }) => original_input.conv_history,
      },
      answerChain,
    ]);

    const response = await chain.invoke({
      question: input,
      conv_history: convHistory.current,
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
      language: "Chinese",
    });

    return response;
  }

  const setup = useCallback(async () => {
    // 初始背景文档获取、分块、向量化、存储
    // await chunk_split_embedding_store();

    // 根据input生成standAloneInput，在向量数据库中查找最近解
    // await standalone_retrievel_answer();
    // 序列化纠正原始input
    // await punctuation_grammar_translate(input);  
  }, [])

  useEffect(() => { setup() }, [setup])

  // 对话记录
  const [log, setLog] = useState(["面试官你好，我是李名河，请开始你的提问？"]);

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
      const input = await punctuation_grammar_translate(_inputValue);
      const res = await standalone_retrievel_answer(input);
      addLog(res);
    } catch (error) {
      addLog("抱歉，好像发生了一些未知错误，请检查一些网络环境或稍后再试。");
    } finally {
      setLoading(false);
    }
  };

  const convHistory = useRef([]);

  useEffect(() => {
    // 在组件挂载时或者其他时机执行滚动到底部的逻辑
    scrollToBottom();
    convHistory.current = formatConvHistory(log);
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
