import React, { useEffect } from 'react';
import ChatbotWidget from '@site/src/components/ChatbotWidget/ChatbotWidget';

const LayoutWithChatbot = (props) => {
  useEffect(() => {
    // Initialize the chatbot when the layout loads
    console.log('Chatbot initialized');
  }, []);

  return (
    <>
      {props.children}
      <ChatbotWidget />
    </>
  );
};

export default LayoutWithChatbot;