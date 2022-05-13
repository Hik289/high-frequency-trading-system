package com.boot.controller;

import java.util.ArrayList;
import java.util.List;

import javax.persistence.criteria.CriteriaBuilder;
import javax.persistence.criteria.CriteriaQuery;
import javax.persistence.criteria.Predicate;
import javax.persistence.criteria.Root;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.boot.entity.User;
import com.boot.repository.UserRepository;
import com.boot.util.PageRequestHelper;
import com.boot.util.StringUtil;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
import com.boot.vo.TokenHandler;
/**
 * 用户控制器
 */
@Controller
@RequestMapping("/private/user")
public class UserController {
	@Autowired
	private UserRepository userRepository;
	
	@ResponseBody
	@RequestMapping("/list")
	public  BaseMessage<?>  userList(String username, HttpServletRequest request, HttpServletResponse response) {
		try {
			Pageable pageable = PageRequestHelper.buildPageRequest(request, null);
			Specification<User> spec = new Specification<User>() {
				private static final long serialVersionUID = 3348042767886904924L;
				@Override
				public Predicate toPredicate(Root<User> root, CriteriaQuery<?> query, CriteriaBuilder cb) {
					List<Predicate> list = new ArrayList<Predicate>();
					if (StringUtil.isNotBlank(username)) {
						list.add(cb.like(root.get("username"), "%" + username + "%"));
					}
					Predicate[] p2 = new Predicate[list.size()];
					query.where(cb.and(list.toArray(p2)));
					return query.getRestriction();
				}
			};
			Page<User> pageList = userRepository.findAll(spec, pageable);
			return MessageHandler.createSuccessVo(pageList.getContent(),"operate successfully",
					(int) pageList.getTotalElements());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	/**
	 * 删除
	 * @param id
	 * @param request
	 * @param response
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/dell")
	public  BaseMessage<?>  dell(Integer id, HttpServletRequest request, HttpServletResponse response) {
		try {
			userRepository.deleteById(id);
			return MessageHandler.createSuccessVo("operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	/**
	 * 保存
	 * @param u
	 * @param request
	 * @param response
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/save")
	public  BaseMessage<?>  save(User u, HttpServletRequest request, HttpServletResponse response) {
		try {
			userRepository.save(u);
			return MessageHandler.createSuccessVo("operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	/**
	 * 修改密码 
	 */
	@ResponseBody
	@RequestMapping("/updatePassword")
	public  BaseMessage<?> updatePassword(String old_password,String new_password,HttpServletRequest request, HttpServletResponse response) {
		try {
			User u = userRepository.findById(TokenHandler.getBusinessId()).get();
			if(u==null) return MessageHandler.createFailedVo("操作失败，用户不存在");
			
			if(!u.getPassword().equals(old_password)) return MessageHandler.createFailedVo("操作失败，旧的密码不正确");
			u.setPassword(new_password);
			userRepository.save(u);
			return MessageHandler.createSuccessVo("operate successfully，请重新登录");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	/**
	 * 信息
	 * @param request
	 * @param response
	 * @param model
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/info")
	public BaseMessage<?> info(HttpServletRequest request, HttpServletResponse response,ModelMap model) {
		User user = userRepository.findById(TokenHandler.getBusinessId()).get();
		return MessageHandler.createSuccessVo(user,"查询成功");
	}
	/**
	 * 更新
	 * @param request
	 * @param response
	 * @param model
	 * @return
	 */
	@ResponseBody
	@RequestMapping("/infoupdate")
	public BaseMessage<?> infoUpdate(User user) {
		User user2 = userRepository.findById(TokenHandler.getBusinessId()).get();
		user2.setName(user.getName());
		user2.setBio(user.getBio());
		userRepository.save(user2);
		return MessageHandler.createSuccessVo("operate successfully");
	}
	/**
	 * 信息保存
	 * @param user
	 * @param request
	 * @param response
	 * @param model
	 * @return
	 */
	@RequestMapping("/info/save")
	@ResponseBody
	public BaseMessage<?> infoSave(User user,HttpServletRequest request, HttpServletResponse response,ModelMap model) {
		try {
			User u = userRepository.findById(TokenHandler.getBusinessId()).get();
			u.setName(user.getName());
			u.setSex(user.getSex());
			u.setAge(user.getAge());
			u.setPhone(user.getPhone());
			u.setBio(user.getBio());
			userRepository.save(u);
			return MessageHandler.createSuccessVo("operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
}
