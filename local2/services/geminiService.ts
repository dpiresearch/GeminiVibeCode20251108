import { Language } from '../types';

const API_BASE_URL = 'http://localhost:3001/api';

export async function translateCode(
  sourceLanguage: Language,
  targetLanguage: Language,
  sourceCode: string,
  feedback?: { previousCode: string; userFeedback: string }
): Promise<string> {
  try {
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sourceLanguage,
        targetLanguage,
        sourceCode,
        feedback,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    return data.translatedCode;
  } catch (error) {
    console.error('Translation API call failed:', error);
    if (error instanceof Error) {
      throw new Error(`Failed to translate code: ${error.message}`);
    }
    throw new Error('Failed to translate code. Please check if the server is running and Ollama is available.');
  }
}

export async function verifyCode(
  language: Language,
  code: string,
  feedback?: {
    previousScript: string;
    userFeedback: string;
    previousError: string;
  }
): Promise<{
  result: string;
  packagedScript: string;
  executionError: string;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        language,
        code,
        feedback,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    return {
      result: data.result,
      packagedScript: data.packagedScript || '',
      executionError: data.executionError || ''
    };
  } catch (error) {
    console.error('Verification API call failed:', error);
    if (error instanceof Error) {
      throw new Error(`Failed to verify code: ${error.message}`);
    }
    throw new Error('Failed to verify code. Please check if the server is running and SSH connection is available.');
  }
}

export async function testSSHConnection(): Promise<{
  success: boolean;
  message: string;
  logs: string;
  error?: string;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/test-ssh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('SSH test API call failed:', error);
    return {
      success: false,
      message: 'Failed to connect to server',
      logs: '',
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}