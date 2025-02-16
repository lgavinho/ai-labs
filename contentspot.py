import requests
from streamlit.logger import get_logger
import settings

logger = get_logger(__name__)

class ContentSpotService:

    def __init__(self):
        self.base_url = settings.CONTENT_SPOT_BASE_URL
        self.headers = {
            "Midiacode-Applabel": "midiacode",
            "Accept-Language": "pt-BR"
        }
    
    def get_content(self, code: str) -> dict:
        """
        Retrieves content from ContentSpot API.
        
        Args:
            code (str): The content code to retrieve
            
        Returns:
            dict: The content data or None if request fails
        """
        try:
            querystring = {"code": code}
            url = f"{self.base_url}/content/"
            logger.info(f"GET {url}?{querystring}")
            response = requests.get(
                url,
                headers=self.headers,
                params=querystring
            )
            
            if response.status_code == 200:
                logger.info("Content retrieved successfully")
                return response.json()
            else:
                logger.error("Failed to get content. Status code: %d", response.status_code)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error("Error fetching content: %s", str(e))
            return None
            
    def get_content_by_codes(self, codes: list[str]) -> list:
        """
        Retrieves multiple contents from ContentSpot API.
        
        Args:
            codes (list[str]): List of content codes to retrieve
            
        Returns:
            list: List of content data
        """
        results = []
        for code in codes:
            content = self.get_content(code)
            if content:
                results.append(content)
        return results


# # Example usage
# service = ContentSpotService()
# # Get single content
# content = service.get_content("zh9gTW")
# print(content)
