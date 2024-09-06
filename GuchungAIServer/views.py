from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''


class AIColumnsView(APIView):
    def post(self, request):
        try:
            text = request.data.get('text')

            messages = [
                {"role": "system", "content": (
                    "You are a skilled data analyst. Extract and return the most relevant column names from the provided text. "
                    "Ensure that the column names are concise, relevant, and free of typographical errors. "
                    "The column names should be in Korean and should match the data context exactly."
                )},
                {"role": "user", "content": f"{text}"}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1
            )

            response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            return Response({'response': response_text}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AIResponseView(APIView):
    def post(self, request):
        try:
            text = request.data.get('text')
            columns = request.data.get('columns')

            messages = [
                {"role": "system", "content": (
                    "You are an AI that extracts structured information from text. "
                    "Extract the data into a table format with the following columns. "
                    "Ensure to include all columns and rows. If any data is missing, use '-' to indicate it. "
                    "Make sure that bank account details, such as 'Account Number' and 'Bank Name', are separated correctly. "
                    "If there's a classification and date, please put it in"
                    "Do not merge these columns and ensure consistency across all rows. "
                    "Use '-' if any column value is missing or not available."
                )},
                {"role": "user", "content": (
                        "Here is the text with information that needs to be converted into a table. "
                        "The table should have the following columns: " + ', '.join(columns) + ".\n\n" + text
                )}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1
            )

            response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            return Response({'response': response_text}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)