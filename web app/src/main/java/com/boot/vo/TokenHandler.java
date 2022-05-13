package com.boot.vo;

import java.net.URLDecoder;
import java.util.Date;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;

import org.springframework.web.context.request.RequestAttributes;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import com.alibaba.fastjson.JSON;
import com.boot.intecepter.Intecepter;
import com.boot.util.AESUtil;
import com.boot.util.StringUtil;

public class TokenHandler {
	private static final String KEY = "7099557480123456";

	public static String create(Integer businessId, Integer seriaNo, String username, String name, String userId) throws Exception {
		return format(new Token(businessId, seriaNo, username, name, new Date().getTime(),userId));
	}

	/**
	 * 直接根据businessid创建
	 */
	public static String create(Integer businessId, Integer seriaNo,String userId) throws Exception {
		return format(new Token(businessId, seriaNo, null, null, new Date().getTime(),userId));
	}

	/**
	 * token对象转字符串
	 */
	public static String format(Token token) throws Exception {
		return AESUtil.encode(JSON.toJSONString(token), KEY);
	}

	/**
	 * 字符串转token对象
	 */
	public static Token parse(String token) throws Exception {
		if (StringUtil.isBlank(token)) return null;
		return JSON.parseObject(AESUtil.decode(token, KEY), Token.class);
	}

	/**
	 * 获取请求的Token
	 * 
	 * @return
	 * @throws Exception
	 */
	public static Token getBusinesser() throws Exception {
		RequestAttributes ra = RequestContextHolder.getRequestAttributes();
		ServletRequestAttributes sra = (ServletRequestAttributes) ra;
		HttpServletRequest request = sra.getRequest();
		String token = (String) request.getAttribute(Intecepter.TOKEN_KEY);
		if (StringUtil.isBlank(token)) {
			token = request.getParameter(Intecepter.TOKEN_KEY);
		}
		if (StringUtil.isBlank(token)) {
			Cookie[] cookies = request.getCookies();
			if (cookies != null) {
				for (Cookie cookie : cookies) {
					if (Intecepter.TOKEN_KEY.equals(cookie.getName())) {
						token = URLDecoder.decode(cookie.getValue(), "UTF-8");
					}
				}
			} else
				return null;
		}
		if (StringUtil.isBlank(token))
			return null;
		return TokenHandler.parse(token);
	}

	/**
	 * 获取请求request对象
	 * 
	 * @return
	 */
	public static HttpServletRequest getRequest() {
		RequestAttributes ra = RequestContextHolder.getRequestAttributes();
		ServletRequestAttributes sra = (ServletRequestAttributes) ra;
		return sra.getRequest();
	}

	/**
	 * 根据请求的Token解析出BusinessId
	 * 
	 * @return
	 * @throws Exception
	 */
	public static Integer getBusinessId() {
		Token token = null;
		try {
			token = getBusinesser();
		} catch (Exception e) {
			// 捕捉异常
			e.printStackTrace();
		}
		if (token != null) {
			return token.getBusinessId();
		}
		return 0;
	}

	/**
	 * 获取请求的Token
	 * 
	 * @return
	 * @throws Exception
	 */
	public static String getCookieUsername() {
		RequestAttributes ra = RequestContextHolder.getRequestAttributes();
		ServletRequestAttributes sra = (ServletRequestAttributes) ra;
		HttpServletRequest request = sra.getRequest();
		String token = (String) request.getAttribute("username");
		if (StringUtil.isBlank(token)) {
			token = request.getParameter(Intecepter.TOKEN_KEY);
		}

		if (StringUtil.isBlank(token)) {
			Cookie[] cookies = request.getCookies();
			if (cookies != null) {
				for (Cookie cookie : cookies) {
					if ("username".equals(cookie.getName())) {
						token = cookie.getValue();
					}
				}
			}
		}
		if (StringUtil.isBlank(token))
			return "";
		return token;
	}
	/**
	 * 获取请求的Token String
	 * @return
	 * @throws Exception
	 */
	public static String getToken() throws Exception {
		RequestAttributes ra = RequestContextHolder.getRequestAttributes();
		ServletRequestAttributes sra = (ServletRequestAttributes) ra;
		HttpServletRequest request = sra.getRequest();
		String token = (String) request.getAttribute(Intecepter.TOKEN_KEY);
		if (StringUtil.isBlank(token)) {
			token=request.getParameter(Intecepter.TOKEN_KEY);
		}
		
		if (StringUtil.isBlank(token)) {
			Cookie[] cookies = request.getCookies();
			if (cookies != null) {
				for (Cookie cookie : cookies) {
					if (Intecepter.TOKEN_KEY.equals(cookie.getName())) {
						token = URLDecoder.decode(cookie.getValue(), "UTF-8");
					}
				}
			} else
				return null;
		}
		if (StringUtil.isBlank(token))
			return null;
		return token;
	}
}
