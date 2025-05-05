from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .main import agent_executor, format_instructions  # import your logic


class ParseInputView(APIView):
    def post(self, request):
        user_query = request.data.get("input", "")
        if not user_query:
            return Response(
                {"error": "No input provided."}, status=status.HTTP_400_BAD_REQUEST
            )
        response = agent_executor.invoke(
            {"input": user_query, "format_instructions": format_instructions}
        )
        final_output = response.get("output", response)
        return Response(final_output)
