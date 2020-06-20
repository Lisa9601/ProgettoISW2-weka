package main.java;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class JSONReader {
	
	
    public String readAll(Reader rd) throws IOException {
    	StringBuilder sb = new StringBuilder();
    	int cp;
    	while ((cp = rd.read()) != -1) {
    		sb.append((char) cp);
    	}
    	return sb.toString();
    }

   
    public JSONArray readJsonArrayFromUrl(String url, String token) throws IOException, JSONException {
    	URL url2 = new URL(url);
    	HttpURLConnection urlConnection = (HttpURLConnection)  url2.openConnection();
   
    	//Setting the requirements to access the github api
    	urlConnection.setRequestProperty("Accept", "application/vnd.github.cloak-preview");
    	urlConnection.setRequestProperty("Authorization", "token "+token);

    	BufferedReader rd = new BufferedReader(new InputStreamReader(urlConnection.getInputStream(), StandardCharsets.UTF_8));
    	String jsonText = readAll(rd);
    	JSONArray json = new JSONArray(jsonText);
  
    	urlConnection.disconnect();
   
    	return json;
      
    }
   
   
    public JSONObject readJsonFromUrl(String url) throws IOException, JSONException {
    	InputStream is = new URL(url).openStream();
	      
    	BufferedReader rd = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
    	String jsonText = readAll(rd);
    	JSONObject json = new JSONObject(jsonText);
	      
    	is.close();
	      
    	return json;
	      
    }
    
    public JSONObject readJsonFromUrl(String url, String token) throws IOException, JSONException {
    	URL url2 = new URL(url);
    	HttpURLConnection urlConnection = (HttpURLConnection)  url2.openConnection();
   
    	//Setting the requirements to access the github api
    	urlConnection.setRequestProperty("Accept", "application/vnd.github.cloak-preview");
    	urlConnection.setRequestProperty("Authorization", "token "+token);
	      
    	BufferedReader rd = new BufferedReader(new InputStreamReader(urlConnection.getInputStream(), StandardCharsets.UTF_8));
    	String jsonText = readAll(rd);
    	JSONObject json = new JSONObject(jsonText);
	      
    	urlConnection.disconnect();
	      
    	return json;
	      
    }

}
