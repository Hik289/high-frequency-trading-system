package com.boot.intecepter;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;

import com.boot.entity.User;
import com.boot.repository.UserRepository;
import com.boot.vo.TokenHandler;

@Component
public class Intecepter extends HandlerInterceptorAdapter {
	public static final String TOKEN_KEY = "token";
	@Autowired
	private UserRepository userRepository;
	@Override
	public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
		boolean result = true;
		System.out.println("=======");
		try {
			Integer id = TokenHandler.getBusinessId();
			User u = null;
			if(id==null||id==0) result= false;
			else {
				u = userRepository.getOne(id);
				if(u==null||u.getId()==null) result = false;
			}
			if(!result)response.sendRedirect("/login"); 
		}catch (Exception e) {
			e.printStackTrace();
			result = false;
			throw new Exception(e.getMessage());
		}
		return result;
	}
}
