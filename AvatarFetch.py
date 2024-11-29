import requests

# GraphQL endpoint for login and fetching avatars
graphql_url = "https://api.take0.ai/api/graphql/"
refresh_token_url = "https://api.take0.ai/api/auth/token-refresh/"

# Fixed user credentials
USERNAME = "dataadmin@aimonk.com"
PASSWORD = "\"Qi9\"\"2aU,GwT^A"

def login():
    """
    Perform login and retrieve access and refresh tokens using fixed credentials.
    """
    login_query = """
    mutation loginFormMutation($input: ObtainTokenMutationInput!) {
      tokenAuth(input: $input) {
        access
        refresh
        otpAuthToken
      }
    }
    """
    variables = {"input": {"email": USERNAME, "password": PASSWORD}}
    response = requests.post(
        graphql_url,
        json={"query": login_query, "variables": variables}
    )
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            raise Exception(f"Login failed: {data['errors']}")
        return data["data"]["tokenAuth"]
    else:
        raise Exception(f"Login request failed with status {response.status_code}, response: {response.text}")

def fetch_avatars(access_token):
    """
    Fetch all avatars using the provided access token.
    """
    avatar_query = """
    query fetchAvatars {
      allAvatar {
        edges {
          node {
            id
            name
            actualId
            isPremium
            poseVideo {
              url
              name
            }
            rank
            thumbnail {
              url
              name
            }
            transparentImage {
              url
              name
            }
          }
        }
      }
    }
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(
        graphql_url,
        json={"query": avatar_query},
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            raise Exception(f"Error fetching avatars: {data['errors']}")
        return data["data"]["allAvatar"]["edges"]
    else:
        raise Exception(f"Avatar request failed with status {response.status_code}")

def refresh_access_token(refresh_token):
    """
    Refresh the access token using the refresh token.
    """
    response = requests.post(
        refresh_token_url,
        data={"refresh": refresh_token}
    )
    if response.status_code == 200:
        return response.json()["access"]
    else:
        raise Exception(f"Token refresh failed with status {response.status_code}")

def get_avatars():
    """
    Login and fetch the list of avatars using fixed credentials.
    """
    try:
        # Step 1: Login and get tokens
        tokens = login()
        access_token = tokens["access"]

        # Step 2: Fetch avatars
        avatars = fetch_avatars(access_token)
        return avatars  # Return the list of avatars

    except Exception as e:
        raise Exception(f"An error occurred while fetching avatars: {e}")

# Example usage
if __name__ == "__main__":
    try:
        avatars = get_avatars()
        print("Avatars fetched successfully:")
        print(avatars[0])  # Print the first avatar
    except Exception as e:
        print(f"An error occurred: {e}")